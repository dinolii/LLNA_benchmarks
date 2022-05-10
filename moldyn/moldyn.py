import subprocess
import statistics
from decimal import *

settings = {
    0: "-DDENSITY=0.02 -DTOLERANCE=5 -DCUTOFF=3.5000",
    10: "-DDENSITY=5 -DTOLERANCE=5 -DCUTOFF=3.5000",
    20: "-DDENSITY=10 -DTOLERANCE=5 -DCUTOFF=3.5000",
    30: "-DDENSITY=15 -DTOLERANCE=5 -DCUTOFF=3.5000",
    40: "-DDENSITY=20 -DTOLERANCE=5 -DCUTOFF=3.5000",
    50: "-DDENSITY=25 -DTOLERANCE=5 -DCUTOFF=3.5000",
    60: "-DDENSITY=0.7235 -DTOLERANCE=1.2 -DCUTOFF=3.5000",
    70: "-DDENSITY=0.6928 -DTOLERANCE=1.2 -DCUTOFF=2.2",
    80: "-DDENSITY=0.801 -DTOLERANCE=1.1 -DCUTOFF=2.2",
    90: "-DDENSITY=0.7975 -DTOLERANCE=1.1 -DCUTOFF=2.2",
    100: "-DDENSITY=0.7 -DTOLERANCE=1.1 -DCUTOFF=2.2",
}


# undef = "-UDENSITY -UTOLERANCE -UCUTOFF"

def run_bench(out, name, extra=[]):
    out.write("Branch Probability,Time")
    for prob in range(0, 101, 10):
        subprocess.run(
            ['gcc', '-mavx2', '-O0', '-w', name + '.c', '-o', name + '.out', '-lm'] + settings[prob].split() + extra)
        timings = subprocess.run(['./{}.out'.format(name)], capture_output=True)
        timing = timings.stdout.decode('utf-8')
        iterations = None
        tmp = []
        for result in timing.splitlines():
            if result == '':
                continue
            entry, iterations = result.split()
            tmp += [Decimal(entry)]
        median = statistics.median(tmp)
        print(prob, median / Decimal(iterations))
        out.write("{},{}".format(prob, median / Decimal(iterations)))

with open('moldyn-scalar.csv', 'w') as out:
    run_bench(out, 'moldyn')

print("finished scalar...")

# with open('moldyn-sse.csv', 'w') as out:
#     run_bench(out, 'moldyn_sse')
#
# print("finished sse...")
#
# with open('moldyn-avx.csv', 'w') as out:
#     run_bench(out, 'moldyn_avx')
#
# print("finished avx...")
