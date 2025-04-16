FROM postgres:15-alpine

COPY src/postgresql/init-db.sh /docker-entrypoint-initdb.d/

RUN chmod +x /docker-entrypoint-initdb.d/init-db.sh
