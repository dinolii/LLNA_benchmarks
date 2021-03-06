TUNE=peak
LABEL=none
NUMBER=519
NAME=lbm_r
SOURCES= lbm.c main.c
EXEBASE=lbm_r
NEED_MATH=yes
BENCHLANG=C

BENCH_FLAGS      = -DSPEC_AUTO_SUPPRESS_OPENMP
CC               = ${tooldir}/clang
CXX              = ${tooldir}/clang++
OS               = unix
PASS1_OPTIMIZE   = -fprofile-generate=pgo_prof.dir
PASS2_OPTIMIZE   = -fprofile-use=pgo_prof.profdata
absolutely_no_locking = 0
abstol           = 1e-07
action           = build
allow_label_override = 0
backup_config    = 1
baseexe          = lbm_r
basepeak         = 0
benchdir         = benchspec
benchmark        = 519.lbm_r
binary           = 
bindir           = exe
builddir         = build
bundleaction     = 
bundlename       = 
calctol          = 1
changedhash      = 0
check_version    = 0
clean_between_builds = no
command_add_redirect = 0
commanderrfile   = speccmds.err
commandexe       = lbm_r_peak.none
commandfile      = speccmds.cmd
commandoutfile   = speccmds.out
commandstdoutfile = speccmds.stdout
comparedir       = compare
compareerrfile   = compare.err
comparefile      = compare.cmd
compareoutfile   = compare.out
comparestdoutfile = compare.stdout
compile_error    = 0
compwhite        = 
configdir        = config
configfile       = bangtian_pgo.cfg
configpath       = /home/labuser/benchmarkcode/src/config/bangtian_pgo.cfg
copies           = 1
current_range    = 
datadir          = data
default_size     = ref
default_submit   = $command
delay            = 0
deletebinaries   = 0
deletework       = 0
dependent_workloads = 0
device           = 
difflines        = 10
dirprot          = 511
discard_power_samples = 0
enable_monitor   = 1
endian           = 12345678
env_vars         = 0
expand_notes     = 0
expid            = 
exthash_bits     = 256
failflags        = 0
fake             = 0
fdo_post1        = ${tooldir}/llvm-profdata merge pgo_prof.dir -o pgo_prof.profdata
fdo_run1         = ${command}
feedback         = 1
flag_url_base    = https://www.spec.org/auto/cpu2017/Docs/benchmarks/flags/
floatcompare     = 
force_monitor    = 0
hostname         = thinkmate-x99
http_proxy       = 
http_timeout     = 30
hw_cpu_name      = Intel Core i7-5820K
hw_disk          = 917 GB  add more disk info here
hw_memory001     = 31.322 GB fixme: If using DDR3, format is:
hw_memory002     = 'N GB (M x N GB nRxn PCn-nnnnnR-n, ECC)'
hw_nchips        = 1
idle_current_range = 
idledelay        = 10
idleduration     = 60
ignore_errors    = 0
ignore_sigint    = 0
ignorecase       = 
info_wrap_columns = 50
inputdir         = input
inputgenerrfile  = inputgen.err
inputgenfile     = inputgen.cmd
inputgenoutfile  = inputgen.out
inputgenstdoutfile = inputgen.stdout
iteration        = -1
iterations       = 3
keeptmp          = 0
label            = none
line_width       = 0
link_input_files = 1
locking          = 1
log              = CPU2017
log_line_width   = 0
log_timestamp    = 0
logname          = /home/labuser/benchmarkcode/src/result/CPU2017.030.log
lognum           = 030
mail_reports     = all
mailcompress     = 0
mailmethod       = smtp
mailport         = 25
mailserver       = 127.0.0.1
mailto           = 
make             = specmake
make_no_clobber  = 0
makefile_template = Makefile.YYYtArGeTYYYspec
makeflags        = 
max_average_uncertainty = 1
max_hum_limit    = 0
max_report_runs  = 3
max_unknown_uncertainty = 1
mean_anyway      = 0
meter_connect_timeout = 30
meter_errors_default = 5
meter_errors_percentage = 5
min_report_runs  = 2
min_temp_limit   = 20
minimize_builddirs = 0
minimize_rundirs = 0
name             = lbm_r
nansupport       = 
need_math        = yes
no_input_handler = close
no_monitor       = 
noratios         = 0
note_preenv      = 0
notes_plat_sysinfo_000 =  Sysinfo program /home/labuser/benchmarkcode/src/bin/sysinfo
notes_plat_sysinfo_005 =  Rev: r5797 of 2017-06-14 96c45e4568ad54c135fd618bcc091c0f
notes_plat_sysinfo_010 =  running on thinkmate-x99 Sun Apr 10 16:18:26 2022
notes_plat_sysinfo_015 = 
notes_plat_sysinfo_020 =  SUT (System Under Test) info as seen by some common utilities.
notes_plat_sysinfo_025 =  For more information on this section, see
notes_plat_sysinfo_030 =     https://www.spec.org/cpu2017/Docs/config.html\#sysinfo
notes_plat_sysinfo_035 = 
notes_plat_sysinfo_040 =  From /proc/cpuinfo
notes_plat_sysinfo_045 =     model name : Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz
notes_plat_sysinfo_050 =        1  "physical id"s (chips)
notes_plat_sysinfo_055 =        12 "processors"
notes_plat_sysinfo_060 =     cores, siblings (Caution: counting these is hw and system dependent. The following
notes_plat_sysinfo_065 =     excerpts from /proc/cpuinfo might not be reliable.  Use with caution.)
notes_plat_sysinfo_070 =        cpu cores : 6
notes_plat_sysinfo_075 =        siblings  : 12
notes_plat_sysinfo_080 =        physical 0: cores 0 1 2 3 4 5
notes_plat_sysinfo_085 = 
notes_plat_sysinfo_090 =  From lscpu:
notes_plat_sysinfo_095 =       Architecture:          x86_64
notes_plat_sysinfo_100 =       CPU op-mode(s):        32-bit, 64-bit
notes_plat_sysinfo_105 =       Byte Order:            Little Endian
notes_plat_sysinfo_110 =       CPU(s):                12
notes_plat_sysinfo_115 =       On-line CPU(s) list:   0-11
notes_plat_sysinfo_120 =       Thread(s) per core:    2
notes_plat_sysinfo_125 =       Core(s) per socket:    6
notes_plat_sysinfo_130 =       Socket(s):             1
notes_plat_sysinfo_135 =       NUMA node(s):          1
notes_plat_sysinfo_140 =       Vendor ID:             GenuineIntel
notes_plat_sysinfo_145 =       CPU family:            6
notes_plat_sysinfo_150 =       Model:                 63
notes_plat_sysinfo_155 =       Model name:            Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz
notes_plat_sysinfo_160 =       Stepping:              2
notes_plat_sysinfo_165 =       CPU MHz:               1199.988
notes_plat_sysinfo_170 =       CPU max MHz:           3600.0000
notes_plat_sysinfo_175 =       CPU min MHz:           1200.0000
notes_plat_sysinfo_180 =       BogoMIPS:              6596.53
notes_plat_sysinfo_185 =       Virtualization:        VT-x
notes_plat_sysinfo_190 =       L1d cache:             32K
notes_plat_sysinfo_195 =       L1i cache:             32K
notes_plat_sysinfo_200 =       L2 cache:              256K
notes_plat_sysinfo_205 =       L3 cache:              15360K
notes_plat_sysinfo_210 =       NUMA node0 CPU(s):     0-11
notes_plat_sysinfo_215 =       Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov
notes_plat_sysinfo_220 =       pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp
notes_plat_sysinfo_225 =       lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf
notes_plat_sysinfo_230 =       eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr
notes_plat_sysinfo_235 =       pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx
notes_plat_sysinfo_240 =       f16c rdrand lahf_lm abm epb tpr_shadow vnmi flexpriority ept vpid fsgsbase
notes_plat_sysinfo_245 =       tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm xsaveopt cqm_llc cqm_occup_llc
notes_plat_sysinfo_250 =       dtherm ida arat pln pts
notes_plat_sysinfo_255 = 
notes_plat_sysinfo_260 =  /proc/cpuinfo cache data
notes_plat_sysinfo_265 =     cache size : 15360 KB
notes_plat_sysinfo_270 = 
notes_plat_sysinfo_275 =  From numactl --hardware  WARNING: a numactl 'node' might or might not correspond to a
notes_plat_sysinfo_280 =  physical chip.
notes_plat_sysinfo_285 =    available: 1 nodes (0)
notes_plat_sysinfo_290 =    node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11
notes_plat_sysinfo_295 =    node 0 size: 32073 MB
notes_plat_sysinfo_300 =    node 0 free: 471 MB
notes_plat_sysinfo_305 =    node distances:
notes_plat_sysinfo_310 =    node   0
notes_plat_sysinfo_315 =      0:  10
notes_plat_sysinfo_320 = 
notes_plat_sysinfo_325 =  From /proc/meminfo
notes_plat_sysinfo_330 =     MemTotal:       32843496 kB
notes_plat_sysinfo_335 =     HugePages_Total:       0
notes_plat_sysinfo_340 =     Hugepagesize:       2048 kB
notes_plat_sysinfo_345 = 
notes_plat_sysinfo_350 =  /usr/bin/lsb_release -d
notes_plat_sysinfo_355 =     Ubuntu 16.04.1 LTS
notes_plat_sysinfo_360 = 
notes_plat_sysinfo_365 =  From /etc/*release* /etc/*version*
notes_plat_sysinfo_370 =     debian_version: stretch/sid
notes_plat_sysinfo_375 =     os-release:
notes_plat_sysinfo_380 =        NAME="Ubuntu"
notes_plat_sysinfo_385 =        VERSION="16.04.1 LTS (Xenial Xerus)"
notes_plat_sysinfo_390 =        ID=ubuntu
notes_plat_sysinfo_395 =        ID_LIKE=debian
notes_plat_sysinfo_400 =        PRETTY_NAME="Ubuntu 16.04.1 LTS"
notes_plat_sysinfo_405 =        VERSION_ID="16.04"
notes_plat_sysinfo_410 =        HOME_URL="http://www.ubuntu.com/"
notes_plat_sysinfo_415 =        SUPPORT_URL="http://help.ubuntu.com/"
notes_plat_sysinfo_420 = 
notes_plat_sysinfo_425 =  uname -a:
notes_plat_sysinfo_430 =     Linux thinkmate-x99 4.4.0-53-generic \#74-Ubuntu SMP Fri Dec 2 15:59:10 UTC 2016 x86_64
notes_plat_sysinfo_435 =     x86_64 x86_64 GNU/Linux
notes_plat_sysinfo_440 = 
notes_plat_sysinfo_445 =  run-level 5 Apr 4 20:46
notes_plat_sysinfo_450 = 
notes_plat_sysinfo_455 =  SPEC is set to: /home/labuser/benchmarkcode/src
notes_plat_sysinfo_460 =     Filesystem     Type  Size  Used Avail Use% Mounted on
notes_plat_sysinfo_465 =     /dev/sdb1      ext4  917G  530G  341G  61% /home
notes_plat_sysinfo_470 = 
notes_plat_sysinfo_475 =  Additional information from dmidecode follows.  WARNING: Use caution when you interpret
notes_plat_sysinfo_480 =  this section. The 'dmidecode' program reads system data which is "intended to allow
notes_plat_sysinfo_485 =  hardware to be accurately determined", but the intent may not be met, as there are
notes_plat_sysinfo_490 =  frequent changes to hardware, firmware, and the "DMTF SMBIOS" standard.
notes_plat_sysinfo_495 = 
notes_plat_sysinfo_500 =  (End of data from sysinfo program)
notes_wrap_columns = 0
notes_wrap_indent =   
num              = 519
obiwan           = 
os_exe_ext       = 
output_format    = default
output_root      = 
outputdir        = output
parallel_test    = 0
parallel_test_submit = 0
parallel_test_workloads = 
path             = /home/labuser/benchmarkcode/src/benchspec/CPU/519.lbm_r
plain_train      = 1
platform         = 
power            = 0
preenv           = 1
prefix           = 
prepared_by      = labuser  (is never output, only tags rawfile)
ranks            = 1
rawhash_bits     = 256
rebuild          = 0
reftime          = reftime
reltol           = 
reportable       = 0
resultdir        = result
review           = 0
run              = all
runcpu           = /home/labuser/benchmarkcode/src/bin/harness/runcpu --config=bangtian_pgo.cfg
rundir           = run
runmode          = rate
safe_eval        = 1
save_build_files = 
section_specifier_fatal = 1
setprocgroup     = 1
setup_error      = 0
sigint           = 2
size             = refrate
size_class       = ref
skipabstol       = 
skipobiwan       = 
skipreltol       = 
skiptol          = 
smarttune        = peak
specdiff         = specdiff
specrun          = specinvoke
srcalt           = 
srcdir           = src
srcsource        = /home/labuser/benchmarkcode/src/benchspec/CPU/519.lbm_r/src
stagger          = 10
strict_rundir_verify = 0
sw_file          = ext4
sw_os001         = Ubuntu 16.04.1 LTS
sw_os002         = 4.4.0-53-generic
sw_state         = Run level 5 (add definition here)
sysinfo_hash_bits = 256
sysinfo_program  = specperl /home/labuser/benchmarkcode/src/bin/sysinfo
sysinfo_program_hash = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
table            = 1
teeout           = 0
test_date        = Apr-2022
threads          = 1
tooldir          = /home/labuser/ssd/Code-Backup/llvm-project/build/bin/
top              = /home/labuser/benchmarkcode/src
train_single_thread = 0
train_with       = train
tune             = peak
uid              = 1000
unbuffer         = 1
uncertainty_exception = 5
update           = 0
update_url       = http://www.spec.org/auto/cpu2017/updates/
use_submit_for_compare = 0
use_submit_for_speed = 0
username         = labuser
verbose          = 5
verify_binaries  = 1
version          = 0.905000
version_url      = http://www.spec.org/auto/cpu2017/devel_version
voltage_range    = 
worklist         = list
OUTPUT_RMFILES   = lbm.out
