FROM postgres:15-alpine\n\nCOPY src/postgresql/init-db.sh /docker-entrypoint-initdb.d/\nRUN chmod +x /docker-entrypoint-initdb.d/init-db.sh
