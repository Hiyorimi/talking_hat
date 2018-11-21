# Preparations

Requires docker or docker-ce installed

# Limitations

Works with cyrillic & english alphabets.

# Building and running

```bash
docker build -t talking_hat .
# docker rm talking_hat if container was previously created
docker run --name talking_hat -p 5000:5000 talking_hat
```

# Deployment 

```bash
docker run --name talking_hat -p 5000:5000 -d talking_hat
```

# Testing

```bash
curl http://0.0.0.0:5000/predict -X POST -d '{"fullname": "Гарри Поттер"}' -H 'Content-type: application/json'
```

# Load Testing

```bash
docker run --rm --read-only -v `pwd`:`pwd` -w `pwd` jordi/ab -T application/json -p data.json -v 2 -c 100 -n 1000 http://0.0.0.0:5000/predict
# In case Apache Benchmark installed:
ab -p data.json -T application/json -c 100 -n 1000 http://0.0.0.0:5000/predict
```

# Sample load testing results

Concurrncy was set to 100 to test on local machine:

```bash
$ ab -p data.json -T application/json -c 100 -n 10000 http://0.0.0.0:5000/predict
This is ApacheBench, Version 2.3 <$Revision: 1807734 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 0.0.0.0 (be patient)
Completed 1000 requests
Completed 2000 requests
Completed 3000 requests
Completed 4000 requests
Completed 5000 requests
Completed 6000 requests
Completed 7000 requests
Completed 8000 requests
Completed 9000 requests
Completed 10000 requests
Finished 10000 requests


Server Software:        Werkzeug/0.14.1
Server Hostname:        0.0.0.0
Server Port:            5000

Document Path:          /predict
Document Length:        141 bytes

Concurrency Level:      100
Time taken for tests:   248.575 seconds
Complete requests:      10000
Failed requests:        10
   (Connect: 0, Receive: 0, Length: 10, Exceptions: 0)
Total transferred:      2879992 bytes
Total body sent:        1800000
HTML transferred:       1409992 bytes
Requests per second:    40.23 [#/sec] (mean)
Time per request:       2485.747 [ms] (mean)
Time per request:       24.857 [ms] (mean, across all concurrent requests)
Transfer rate:          11.31 [Kbytes/sec] received
                        7.07 kb/s sent
                        18.39 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.3      0       5
Processing:    58 2479 991.2   2294    8731
Waiting:       53 2477 991.2   2293    8730
Total:         63 2479 991.1   2295    8731

Percentage of the requests served within a certain time (ms)
  50%   2295
  66%   2699
  75%   2982
  80%   3179
  90%   3810
  95%   4390
  98%   5092
  99%   5617
 100%   8731 (longest request)
```

