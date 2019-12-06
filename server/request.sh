#!/bin/bash
curl -X POST -H "Content-Type: application/json" -d '{"texts":["이 영화 죽여주는데?", "핵강추", "0", "내일은 정말정말 추울꺼야", "ㅈㄴ 조아"], "max_seq_len":20}' http://0.0.0.0:12345/predict