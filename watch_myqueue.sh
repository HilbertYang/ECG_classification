#!/usr/bin/env bash

set -euo pipefail

POLL_INTERVAL="${POLL_INTERVAL:-15}"
JOB_ID="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  ./watch_myqueue.sh                # Watch all jobs currently listed in myqueue
  ./watch_myqueue.sh <job_id>       # Watch a specific job ID

Optional environment variables:
  POLL_INTERVAL=10 ./watch_myqueue.sh <job_id>
EOF
}

if [[ "${JOB_ID}" == "-h" || "${JOB_ID}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v myqueue >/dev/null 2>&1; then
  echo "Error: myqueue command not found." >&2
  exit 1
fi

queue_output="$(myqueue)"

extract_job_ids() {
  awk 'NR > 3 && $1 ~ /^[0-9]+$/ { print $1 }'
}

if [[ -z "${JOB_ID}" ]]; then
  JOB_IDS="$(printf '%s\n' "${queue_output}" | extract_job_ids)"
  if [[ -z "${JOB_IDS}" ]]; then
    echo "No jobs found in myqueue."
    exit 0
  fi
else
  if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "Error: Job ID must be numeric." >&2
    exit 1
  fi
  JOB_IDS="${JOB_ID}"
fi

job_id_list="$(printf '%s ' ${JOB_IDS})"
job_id_list="${job_id_list% }"

echo "Watching job(s): ${job_id_list}"
echo "Polling interval: ${POLL_INTERVAL} seconds"
echo

print_status() {
  local now
  now="$(date '+%F %T')"
  echo "[${now}] Current queue status:"
  myqueue
  echo
}

all_jobs_done() {
  local current_queue
  current_queue="$(myqueue)"

  while read -r id; do
    [[ -z "${id}" ]] && continue
    if printf '%s\n' "${current_queue}" | awk 'NR > 3 && $1 == job { found=1 } END { exit !found }' job="${id}"; then
      return 1
    fi
  done <<< "${JOB_IDS}"

  return 0
}

big_alert() {
  local line
  line="$(printf '=%.0s' $(seq 1 72))"

  for _ in 1 2 3 4 5; do
    printf '\a'
    sleep 0.2
  done

  echo
  echo "${line}"
  echo "                     YOUR JOB HAS FINISHED!"
  echo "                     Job ID: ${job_id_list}"
  echo "${line}"
  echo
}

print_status

while true; do
  if all_jobs_done; then
    big_alert
    exit 0
  fi

  sleep "${POLL_INTERVAL}"
  print_status
done
