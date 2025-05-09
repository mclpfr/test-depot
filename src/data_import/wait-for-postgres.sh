#!/bin/bash

set -e

host="$1"
port="$2"
shift 2
cmd="$@"

until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -p "$port" -U "$POSTGRES_USER" -d "postgres" -c '\q'; do
  >&2 echo "PostgreSQL is not available - waiting..."
  sleep 1
done

>&2 echo "PostgreSQL is ready - executing command"
exec $cmd 