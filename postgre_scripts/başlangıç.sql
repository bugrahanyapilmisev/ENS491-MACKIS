-- Database: bitirme

-- DROP DATABASE IF EXISTS bitirme;

CREATE DATABASE bitirme
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'English_Turks & Caicos Islands.1252'
    LC_CTYPE = 'English_Turks & Caicos Islands.1252'
    LOCALE_PROVIDER = 'libc'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;


CREATE DATABASE subot_db

CREATE ROLE bugra   LOGIN PASSWORD 'bugra123';
CREATE ROLE berke   LOGIN PASSWORD 'berke123';
CREATE ROLE ahmet   LOGIN PASSWORD 'ahmet123';

GRANT CONNECT ON DATABASE subot_db TO bugra, berke, ahmet;


\c subot_db