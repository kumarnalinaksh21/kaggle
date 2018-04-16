set -x
set -e
LOG="log.info"
exec &> >(tee -a "$LOG")
python t.py 
