import sys
import argparse
import logging
import asyncio

message_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=message_format, stream=sys.stderr, level=logging.INFO)

import time
import argparse
import re
import torch
import sherpa_onnx

# Needed for loading the speaker change detection model
from pytorch_lightning.utilities import argparse_utils
setattr(argparse_utils, "_gpus_arg_default", lambda x: 0)

from asr import TurnDecoder, transcribe_audio
from lid import LanguageFilter
from online_scd.model import SCDModel
from confidence import confidence_filter
from presenters import *
from vad import SpeechSegmentGenerator
from turn import TurnGenerator
import utils
import gc
import tracemalloc 



def process_result(result):
    #result = unk_decoder.post_process(result)    
    text = ""
    if "result" in result:
        result_words = []
        for word in result["result"]:
            if word["word"] in ",.!?" and len(result_words) > 0:
                result_words[-1]["word"] += word["word"]
            else:
                result_words.append(word)
        result["result"] = result_words
        #text = " ".join([wi["word"] for wi in result["result"]])

        #text = compound_reconstructor.post_process(text)
        #text = ray.get(remote_words2numbers.post_process.remote(text))
        #text = ray.get(remote_punctuate.post_process.remote(text))           
        #result = utils.reconstruct_full_result(result, text)
        #result = confidence_filter(result)
        return result
    else:
        return result

def main(args):
    
    if args.youtube_caption_url is not None:
        presenter = YoutubeLivePresenter(captions_url=args.youtube_caption_url)
    elif args.fab_speechinterface_url is not None:
        presenter = FabLiveWordByWordPresenter(fab_speech_iterface_url=args.fab_speechinterface_url)
    elif args.fab_bcast_url is not None:
        presenter = FabBcastWordByWordPresenter(fab_bcast_url=args.fab_bcast_url)
    elif args.zoom_caption_url is not None:
        presenter = ZoomPresenter(captions_url=args.zoom_caption_url)
    else:
        presenter = WordByWordPresenter(args.word_output_file, word_delay_secs=args.word_output_delay)
        #presenter = TerminalPresenter()
    
    scd_model = SCDModel.load_from_checkpoint("models/online-speaker-change-detector/checkpoints/epoch=102.ckpt")
    sherpa_model = create_sherpa_model()


    speech_segment_generator = SpeechSegmentGenerator(args.input_file)
    language_filter = LanguageFilter()        
    
    def main_loop():
        for speech_segment in speech_segment_generator.speech_segments():
            presenter.segment_start()
            
            speech_segment_start_time = speech_segment.start_sample / 16000

            turn_generator = TurnGenerator(scd_model, speech_segment)        
            for i, turn in enumerate(turn_generator.turns()):
                if i > 0:
                    presenter.new_turn()
                turn_start_time = (speech_segment.start_sample + turn.start_sample) / 16000                
                
                turn_decoder = TurnDecoder(sherpa_model, language_filter.filter(turn.chunks()))            
                for res in turn_decoder.decode_results():
                    if "result" in res:
                        processed_res = process_result(res)
                        #processed_res = res
                        if res["final"]:
                            presenter.final_result(processed_res["result"])
                        else:
                            presenter.partial_result(processed_res["result"])
            presenter.segment_end()   
            gc.collect()   

    main_loop()        

def create_sherpa_model():
    """Create and return sherpa-onnx recognizer."""
    return sherpa_onnx.OnlineRecognizer(
        tokens="models/sherpa/tokens.txt",
        encoder="models/sherpa/encoder.onnx",
        decoder="models/sherpa/decoder.onnx",
        joiner="models/sherpa/joiner.onnx",
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=5.0,
        rule2_min_trailing_silence=2.0,
        rule3_min_utterance_length=300,
        decoding_method="modified_beam_search",
        max_feature_vectors=1000,
    )


def wyoming_main(args):
    """Run Wyoming protocol server for Home Assistant integration."""
    from functools import partial
    from wyoming_handler import run_wyoming_server

    logging.info("Starting kiirkirjutaja in Wyoming server mode")
    logging.info(f"Loading sherpa model...")

    sherpa_model = create_sherpa_model()

    logging.info(f"Model loaded. Starting server on {args.wyoming_uri}")

    # Create transcription function with model bound
    def transcribe_func(audio_bytes: bytes) -> str:
        return transcribe_audio(sherpa_model, audio_bytes)

    # Run async server
    asyncio.run(run_wyoming_server(args.wyoming_uri, transcribe_func))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Kiirkirjutaja - Estonian real-time speech recognition"
    )

    # Wyoming mode (Home Assistant)
    parser.add_argument(
        '--wyoming-uri',
        type=str,
        help="Wyoming server URI, e.g., tcp://0.0.0.0:10300"
    )

    # Legacy mode arguments
    parser.add_argument('--youtube-caption-url', type=str)
    parser.add_argument('--fab-speechinterface-url', type=str)
    parser.add_argument('--fab-bcast-url', type=str)
    parser.add_argument('--zoom-caption-url', type=str)
    parser.add_argument('--word-output-file', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--word-output-delay', default=0.0, type=float,
                        help="Words are not outputted before that many seconds have passed since their actual start")
    parser.add_argument('input_file', nargs='?', default=None,
                        help="Input audio file or '-' for stdin")

    args = parser.parse_args()

    # Choose mode based on arguments
    if args.wyoming_uri:
        wyoming_main(args)
    elif args.input_file:
        # Legacy file/stdin mode - needs ray
        import ray
        ray.init(num_cpus=4)
        main(args)
    else:
        parser.print_help()
        sys.exit(1)
    
