CLEAR SCREEN;

-- Clear out tables
DROP TABLE programmers CASCADE CONSTRAINTS;
DROP TABLE tasks CASCADE CONSTRAINTS;
DROP TABLE languages CASCADE CONSTRAINTS;
DROP TABLE completed_tasks CASCADE CONSTRAINTS;
DROP TABLE proficient_in CASCADE CONSTRAINTS;

-- Programmers(ID, date of birth, job title, salary, gender, name, supervisor)
CREATE TABLE programmers (
	id NUMBER(6) PRIMARY KEY,
	date_of_birth NUMBER,
	job_title VARCHAR2(15 CHAR),
	salary NUMBER,
	gender VARCHAR2(8 CHAR),
	name VARCHAR2(15 CHAR),
	supervisor VARCHAR2(15 CHAR)
);

-- Programming Tasks(task name, completion date, start date, target)
CREATE TABLE tasks (
	task_name VARCHAR2(15 CHAR) PRIMARY KEY, 
	completion_date NUMBER, 
	start_date NUMBER, 
	target_date NUMBER
);

-- Programming Language(language name, compiled or interpreted, examination suit, task name)
CREATE TABLE languages (
	language_name VARCHAR2(15 CHAR) PRIMARY KEY, 
	compiled_interpreted VARCHAR2(15 CHAR), 
	examination_suit VARCHAR2(50 CHAR), 
	task_name VARCHAR2(15 CHAR) REFERENCES tasks(task_name)
);

-- Completed Tasks(programmers, days worked, languages)
CREATE TABLE completed_tasks (
	programmer_id NUMBER(6) REFERENCES programmers(id), 
	days_worked NUMBER, 
	languages VARCHAR2(50 CHAR) REFERENCES languages(language_name)
);

-- Proficient In(proficient date, programmer id, languages name)
CREATE TABLE proficient_in (
	proficient_date NUMBER PRIMARY KEY, 
	programmer_id NUMBER(6) REFERENCES programmers(id),
	languages_name VARCHAR2(15 CHAR) REFERENCES languages(language_name)
);

-- Clear anything out
DELETE completed_tasks;
DELETE proficient_in;
DELETE languages;
DELETE tasks;
DELETE programmers;

-- COLUMN settings
COLUMN column_name HEADING column_heading;

--------------------------------------------------------------------------------------------
-- For each task that is active today (the day the query is executed), find the number
-- of days remaining (i.e., till the target completion date): display the task name, the
-- target completion date, and number of remaining months in ascending order of
-- completion date (the earliest will be first).
--------------------------------------------------------------------------------------------
SELECT task_name, target_date
FROM task t
WHERE t.start_date > 05032018 
ORDER BY target_date ASC;

COLUMN task_name FORMAT a15;
COLUMN target_date FORMAT 99999999;

--------------------------------------------------------------------------------------------
-- Find the IDs and names of all programmers who are assigned to a task that is to
-- terminate some time in the year 2017; present the result in ascending order of
-- programmer ID.
--------------------------------------------------------------------------------------------
SELECT id, name
FROM programmers p, task t
WHERE t.target_date < 12302017 and t.target_date > 01012017
ORDER BY p.id ASC;

COLUMN id FORMAT 999999;
COLUMN name FORMAT a15;

--------------------------------------------------------------------------------------------
-- For each programming language used in at least one active tmask, display its name,
-- the most senior programmer(s) in terms of proficiency, and the date that was
-- achieved.
--------------------------------------------------------------------------------------------
SELECT task_name, MAX(proficient_date)
FROM programmers p, language l, proficient_in pn
WHERE l.task_name IN (SELECT target_date
         		FROM tasks t
			WHERE t.target_date > 05032018);

COLUMN name FORMAT a15;
COLUMN proficient_date FORMAT 999999;

--------------------------------------------------------------------------------------------
-- For each programming language, display the id’s and names of programmer-
-- supervisor pairs whenever both the programmer and his/her supervisor are
-- proficient in that language. Do not list a programmer who is his/her own
-- supervisor.
--------------------------------------------------------------------------------------------
SELECT id, supervisor
FROM programmers p, language l
WHERE p.supervisor <> p.id AND p.id IN (SELECT programmer_id 
					FROM proficient_in pn);

COLUMN id FORMAT 999999;
COLUMN supervisor FORMAT a15;
