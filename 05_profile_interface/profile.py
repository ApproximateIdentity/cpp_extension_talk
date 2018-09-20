from data import Timer

from module import noop


def profiler(payload_size, num_calls):
    payload = [float(i) for i in range(payload_size)]
    timer = Timer()
    for _ in range(num_calls):
        noop(payload)
    return timer.split()

print("Payload", "Calls", "Cps", "Bps", sep="\t")
for payload_size, num_calls in [
        (100, 1000000),
        (1000, 100000),
        (10000, 10000),
        (100000, 1000),
        (1000000, 100)]:
    elapsed_time = profiler(payload_size, num_calls)
    print(payload_size, num_calls, int(num_calls / elapsed_time),
          num_calls * 8 * payload_size / elapsed_time, sep="\t")
