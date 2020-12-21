#!/usr/bin/env python
from __future__ import absolute_import, print_function
import wave
import datetime
import argparse
import io
import logging
import os
import sys
import time
from logging import debug, info
import uuid
import cgi
import audioop
import asyncio
import aiofile
from pprint import pprint
from threading import Thread
from threading import Timer
import threading
from queue import Queue
import time
import logging

import psutil

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

import boto3

import requests
import tornado.ioloop
import tornado.websocket
import tornado.httpserver
import tornado.template
import tornado.web
import webrtcvad
from requests_aws4auth import AWS4Auth
from tornado.web import url
import json
from base64 import b64decode
from requests.packages.urllib3.exceptions import InsecurePlatformWarning
from requests.packages.urllib3.exceptions import SNIMissingWarning
from dotenv import load_dotenv
load_dotenv()

#-------------------------

connected = 0
disconnected = 0

# format = '%(asctime)s: %(message)s'
# logging.basicConfig(format=format, level=logging.DEBUG, datefmt='%H:%M:%S')

# # Only used for record function

# logging.captureWarnings(False)
# requests.packages.urllib3.disable_warnings(InsecurePlatformWarning)
# requests.packages.urllib3.disable_warnings(SNIMissingWarning)

MS_PER_FRAME = 20  # Duration of a frame in ms

# Global variables
conns = {}

# Environment variables (local deployment: .env file)
PORT = os.getenv("PORT") # Do not set as Config Vars for Heroku deployment
REGION = os.getenv("REGION", default = "us-east-1")
TRANSCRIBE_LANGUAGE_CODE = os.getenv("TRANSCRIBE_LANGUAGE_CODE", default = "en-US")

# Derivate sentiment language from transcribe language
SENTIMENT_LANGUAGE = TRANSCRIBE_LANGUAGE_CODE[:2]   # e.g. "en"

DELETE_RECORDING = os.getenv("DELETE_RECORDING", default = True)

#-------------------------------- Transcribe main -----------------------------

class MyEventHandler(TranscriptResultStreamHandler):
    
    def __init__(self, *args):
        super().__init__(*args)
        self.result_holder = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.

        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                self.result_holder.append(alt.transcript)
                # print(alt.transcript)
#--     

async def basic_transcribe(file, transcript, media_sample_rate_hz=8000, language_code=TRANSCRIBE_LANGUAGE_CODE, region=REGION):
    
    client = TranscribeStreamingClient(region=region)

    stream = await client.start_stream_transcription(
        language_code=language_code,
        media_sample_rate_hz=media_sample_rate_hz,
        media_encoding="pcm",
    )

    async def write_chunks():

        async with aiofile.AIOFile(file, 'rb') as afp:
            reader = aiofile.Reader(afp, chunk_size=media_sample_rate_hz * 1.024)
            async for chunk in reader:
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
     
        await stream.input_stream.end_stream()

    handler = MyEventHandler(stream.output_stream)

    await asyncio.gather(write_chunks(), handler.handle_events())

    if handler.result_holder == [] :
        transcript.put('')
    else :
        transcript.put(handler.result_holder[-1])

    return()  

#---------------------- Comprehend main ---------------------------------------

comprehend = boto3.client(service_name='comprehend', region_name=REGION)

#------------------- Core processing classes ----------------------------------

class BufferedPipe(object):
    def __init__(self, max_frames, sink):
        """
        Create a buffer which will call the provided `sink` when full.

        It will call `sink` with the number of frames and the accumulated bytes when it reaches
        `max_buffer_size` frames.
        """
        self.sink = sink
        self.max_frames = max_frames

        self.count = 0
        self.payload = b''

    async def append(self, data, id):
        """ Add another data to the buffer. `data` should be a `bytes` object. """

        self.count += 1
        self.payload += data

        if self.count == self.max_frames:
            self.process(id)

    async def process(self, id):
        """ Process and clear the buffer. """
        await self.sink(self.count, self.payload, id)
        self.count = 0
        self.payload = b''


class TranscribeComprehendProcessor(object):
    def __init__(self, path, rate, clip_min, requestor_id, transcribe_comprehend_url, entity, do_sentiment):
        self.rate = rate
        self.bytes_per_frame = rate/25
        self._path = path
        self.clip_min_frames = clip_min // MS_PER_FRAME
        self.client_id = requestor_id
        self.webhook_url = transcribe_comprehend_url
        self.entity = entity
        self.do_sentiment = do_sentiment

    async def process(self, count, payload, id):
        if count > self.clip_min_frames:  # If the buffer is less than CLIP_MIN_MS, ignore it
            
            self.fn = "{}rec-{}-{}.wav".format('./recordings/', id,
                                          datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
            output = wave.open(self.fn, 'wb')
            output.setparams(
                (1, 2, self.rate, 0, 'NONE', 'not compressed'))
            output.writeframes(payload)
            output.close()
            debug('File written {}'.format(self.fn))
            
            self.q = Queue()
            x = Thread(target=asyncio.run, args=(basic_transcribe(file=self.fn,transcript=self.q), ))
            x.start()

            self.checkqueue = True
            
            while (self.checkqueue):
            	try:
            		self.transcript = self.q.get(False)
            		self.checkqueue = False
            		if (DELETE_RECORDING):
            			os.remove(self.fn)
            		break
            	except:
            		self.transcript = None
            		await asyncio.sleep(1)

            print('Transcript:', self.transcript)
            
            self.q.task_done()
            del(x)

            #-----------------
         
            if self.transcript != '' :
                if (self.do_sentiment) :
                    self.sentiment = comprehend.detect_sentiment(Text=self.transcript, LanguageCode=SENTIMENT_LANGUAGE)
                    
                    self.payload_raw = {
                        "transcript": self.transcript,
                        "entity": self.entity,
                        "sentiment": self.sentiment,
                        "client_id": self.client_id,
                        "service": "AWS Transcribe+Comprehend"
                    }
                else:
                    self.payload_raw = {
                        "transcript": str(self.transcript),
                        "entity": str(self.entity),
                        "client_id": self.client_id,
                        "service": "AWS Transcribe"
                    }

                self.payload_out = json.dumps(self.payload_raw)
                info('payload')
                info(self.payload_out)

                # Posting results back via webhook
                if (self.webhook_url):
                	a = requests.post(self.webhook_url, data=self.payload_out, headers={'Content-Type': 'application/json'})

        else:
            info('Discarding {} frames'.format(str(count)))

    def playback(self, response, id):
        if self.rate == 8000:
            content, _ignore = audioop.ratecv(
                response, 2, 1, 16000, 8000, None)  # Downsample 16Khz to 8Khz
        else:
            content = response
        
        frames = int(len(content) // self.bytes_per_frame)
        print(frames)
        info("Playing {} frames to {}".format(frames, id))
        conn = conns[id]
        pos = int(0)
        for x in range(0, frames + 1):
            newpos = int(pos + self.bytes_per_frame)
            #debug("writing bytes {} to {} to socket for {}".format(pos, newpos, id))
            data = content[pos:newpos]
            conn.write_message(data, binary=True)
            pos = newpos

#-------------------------- Websocket handler ------------------------------------

class WSHandler(tornado.websocket.WebSocketHandler):
    
    def initialize(self):
        # Create a buffer which will call `process` when it is full:
        self.frame_buffer = None
        # Setup the Voice Activity Detector
        self.tick = None
        self.id = uuid.uuid4().hex
        self.vad = webrtcvad.Vad()
        # Level of sensitivity
        self.processor = None
        self.path = None
        self.rate = None  # default to None
        self.silence = 20  # default of 20 frames (400ms)
        conns[self.id] = self

    async def open(self, path):
        # info("client connected")
        global connected
        connected = connected + 1
        print(">>> client connected: ", connected)
        debug(self.request.uri)
        self.path = self.request.uri
        self.tick = 0

    async def on_message(self, message):
        # Check if message is Binary or Text
        if type(message) != str:
            if self.vad.is_speech(message, self.rate):
                debug("SPEECH from {}".format(self.id))
                self.tick = self.silence
                await self.frame_buffer.append(message, self.id)
            else:
                debug("Silence from {} TICK: {}".format(self.id, self.tick))
                self.tick -= 1
                if self.tick == 0:
                    # Force processing and clearing of the buffer
                    await self.frame_buffer.process(self.id)
        else:
            info(message)
            # Here we should be extracting the meta data that was sent and attaching it to the connection object
            data = json.loads(message)
            m_type, m_options = cgi.parse_header(data['content-type'])
            
            self.rate = int(m_options['rate'])
            # info(">>> rate")
            # info(self.rate)            

            clip_min = int(data.get('clip_min', 200))
            clip_max = int(data.get('clip_max', 10000))
            silence_time = int(data.get('silence_time', 400))
            
            sensitivity = int(data.get('sensitivity', 3))
            # info(">>> sensitivity")
            # info(sensitivity)
            
            self.client_id = data.get('client_id', "")
            # info(">>> client_id")
            # info(self.client_id) 

            # Webhook URL for analytics (optional for client app)
            self.webhook_url = data.get('webhook_url', "")
            # info(">>> webhook_url")
            # info(self.webhook_url)

            self.entity = data.get('entity', "")

            self.do_sentiment = data.get('do_sentiment', True)
    
            self.vad.set_mode(sensitivity)
            self.silence = silence_time // MS_PER_FRAME
            self.processor = TranscribeComprehendProcessor(
                self.path, self.rate, clip_min, self.client_id, self.webhook_url, self.entity, self.do_sentiment).process
            self.frame_buffer = BufferedPipe(
                clip_max // MS_PER_FRAME, self.processor)
            self.write_message('ok')

    def on_close(self):
        # Remove the connection from the list of connections
        del conns[self.id]
        # info("client disconnected")

        global disconnected
        disconnected = disconnected + 1
        print("<<< client disconnected: ", disconnected)


#------------------------ Web server basic service check ----------------------        

class PingHandler(tornado.web.RequestHandler):
    # @tornado.web.asynchronous
    async def get(self):
        self.write('ok')
        self.set_header("Content-Type", 'text/plain')
        self.finish()

#------------------------- Main thread -----------------------------------------        

def main(argv=sys.argv[1:]):
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--verbose", action="count")
        args = ap.parse_args(argv)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose != None else logging.INFO,
            format="%(levelname)7s %(message)s",
        )
        print("Logging level is {}".format(logging.getLevelName(logging.getLogger().level)))
        application = tornado.web.Application([
            url(r"/ping", PingHandler),
            url(r"/(.*)", WSHandler)
        ])
        http_server = tornado.httpserver.HTTPServer(application)
        http_server.listen(PORT)
        info("Running on port %s", PORT)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        pass  # Suppress the stack-trace on quit

#----------------------- Start main thread --------------------------------------        

if __name__ == "__main__":
    main()
