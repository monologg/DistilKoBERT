#!/bin/bash
curl -X POST -H "Content-Type: application/json" -d '{"texts":["오늘 날씨 어때?", "내일은 정말정말 추울꺼야"], "max_seq_len":20}' http://0.0.0.0:12345/predict