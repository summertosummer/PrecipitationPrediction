/* ************************************************************************* */
/* (1) Name and Information                                                  */
/* ------------------------------------------------------------------------- */
/* Christian Tuton                                                           */
/* 05-03-18                                                                  */
/*                                                                           */
/* Summary:                                                                  */
/* Creates tables from relational schema, declare specifified procedures and */
/* a single disfunctional trigger, popultes tables with data, displays all   */
/* tables, invokes all procedures, contains code to remove all tables.       */
/* ************************************************************************* */

/* ****************** */
/* (2) Create tables */
/* **************** */
CREATE TABLE Tasks (
    name VARCHAR2(15)       NOT NULL,
    startdate DATE          NOT NULL,
    targetcompletion DATE   NOT NULL,
    PRIMARY KEY (name)
);

CREATE TABLE Languages (
    name VARCHAR2(15)        NOT NULL,
    exam VARCHAR2(15)        NOT NULL,
    compiled CHAR(1)         NOT NULL,
    PRIMARY KEY (name)
);

CREATE TABLE Programmers (
    id NUMBER(6)            NOT NULL,
    fname VARCHAR2(15)      NOT NULL,
    lname VARCHAR2(15)      NOT NULL,
    salary NUMBER(11,2)     NOT NULL,
    dob DATE                NOT NULL,
    gender CHAR(1)          NOT NULL,
    advisor NUMBER(6)      NOT NULL,
    currenttask VARCHAR2(15),   -- Cannot guarentee task after project termination
    PRIMARY KEY (id),
    FOREIGN KEY (advisor) REFERENCES Programmers(id),
    FOREIGN KEY (currenttask) REFERENCES Tasks(name)
);

CREATE TABLE Profeciencies (
    programmer NUMBER(6)    NOT NULL,
    lang VARCHAR2(15)       NOT NULL,
    profdate DATE           NOT NULL,
    PRIMARY KEY (programmer, lang),
    FOREIGN KEY (programmer) REFERENCES Programmers(id),
    FOREIGN KEY (lang) REFERENCES Languages(name)
);

CREATE TABLE TaskLanguages (
    task VARCHAR2(15)      NOT NULL,
    programmer NUMBER(6)   NOT NULL,
    lang VARCHAR2(15)      NOT NULL,
    PRIMARY KEY(task, programmer, lang),
    FOREIGN KEY(task) REFERENCES Tasks(name),
    FOREIGN KEY(programmer) REFERENCES Programmers(id),
    FOREIGN KEY(lang) REFERENCES Languages(name)
);

CREATE TABLE Completedtasks (
    taskname VARCHAR2(15)   NOT NULL,
    PRIMARY KEY (taskname)
);

CREATE TABLE CompletedTaskProgrammers (
    fname VARCHAR2(15)         NOT NULL,
    lname VARCHAR2(15)         NOT NULL,
    task VARCHAR2(15)          NOT NULL,
    daysworked NUMBER(5)        NOT NULL,
    PRIMARY KEY (fname, lname, task),
    FOREIGN KEY (task) REFERENCES CompletedTasks(taskname)
);

CREATE TABLE CompletedTaskLanguages (
    fname VARCHAR2(15)         NOT NULL,
    lname VARCHAR2(15)         NOT NULL,
    task VARCHAR2(15)          NOT NULL,
    lang VARCHAR2(15)          NOT NULL,
    FOREIGN KEY (fname, lname, task) REFERENCES CompletedTaskProgrammers(fname, lname, task)
);

/* *********************** */
/* (3) Procedures/Triggers */
/* *********************** */

/* PROCEDURES */

/* P-1) Terminate Task */
-- Note I am assuming this does not imply project completion, just termination
CREATE OR REPLACE PROCEDURE TerimateTask(tsk in Tasks.name%type)
IS
BEGIN
    DECLARE
        CURSOR c is
        select task from TaskLanguages for update;
        c_rec c%rowtype;
    BEGIN
        -- Delete participants in task
        for c_rec in c loop
            if (c_rec.task = tsk) then
                delete from TaskLanguages
                where current of c;
            end if;
        end loop;
        
        -- Delete task
        delete from Tasks
        where name=tsk;
    END;
END;
/

/* P-2) Retire Language */
CREATE OR REPLACE PROCEDURE RetireLanguage(l in languages.name%type)
IS
BEGIN
    DECLARE
        CURSOR c is
        select lang from profeciencies;
        c_rec c%rowtype;
        isfound boolean;
    BEGIN
        isfound := false;
        IF NOT c%isopen then
            open c;
        END IF;
        fetch c into c_rec;
        while c%found loop
            -- Profeciency found
            if l=c_rec.lang then
                isFound := true;
            end if;
            fetch c into c_rec;
        end loop;
        close c;
        
        if isfound=true then
            dbms_output.put_line('Cannot retire language: There exists profecient programmers');
        else
            delete from languages
            where name=l;
        end if;
    END;
END;
/

/* P-3) Retire Programmer */
CREATE OR REPLACE PROCEDURE RetireProgrammer(prg in programmers.id%type)
IS
BEGIN
    DECLARE
        -- Programmers (remove advisor)
        CURSOR c1 is
        select advisor,id from Programmers;
        -- Task Languages (remove self from active projects)
        CURSOR c2 is
        select programmer from TaskLanguages;
        -- Profeciencies (remove self from profeciencies)
        CURSOR c3 is
        select programmer from profeciencies;
        c1_rec c1%rowtype;
        c2_rec c2%rowtype;
        c3_rec c3%rowtype;
    BEGIN
        -- Change advisor to advise themselves
        IF NOT c1%isopen then
            open c1;
        END IF;
        fetch c1 into c1_rec;
        while c1%found loop
            -- Advisor found
            if prg = c1_rec.advisor then
                update programmers
                set advisor = c1_rec.id
                where prg = c1_rec.id;
            end if;
            fetch c1 into c1_rec;
        end loop;
        close c1;
        
        -- Remove self from active projects
        IF NOT c2%isopen then
            open c2;
        END IF;
        fetch c2 into c2_rec;
        while c2%found loop
            -- Self found
            if c2_rec.programmer = prg then
                delete from TaskLanguages
                where programmer = prg;
            end if;
            fetch c2 into c2_rec;
        end loop;
        close c2;
        
        -- Delete from profeciencies
        IF NOT c3%isopen then
            open c3;
        END IF;
        fetch c3 into c3_rec;
        while c3%found loop
            -- profeciency found
            if c3_rec.programmer = prg then
                delete from Profeciencies
                where programmer = prg;
            end if;
            fetch c3 into c3_rec;
        end loop;
        close c3;
        
        -- Delete programmer
        delete from programmers
        where id = prg;
    END;
END;
/

/* P-4) Display Completed Task */
CREATE OR REPLACE PROCEDURE DisplayCompleted(tsk in completedtasks.taskname%type)
IS
BEGIN
    DECLARE
        CURSOR c is
        select * from CompletedTaskProgrammers for update;
        c_rec c%rowtype;
        hasLooped boolean;
    BEGIN
        hasLooped := false;
        for c_rec in c loop
            -- Participant found
            if (c_rec.task = tsk) then
                hasLooped := true;
                dbms_output.put_line(c_rec.fname || ' ' || c_rec.lname || ', ' || c_rec.daysworked);
            end if;
        end loop;
        
        if hasLooped=false then
            dbms_output.put_line('No programmers found for project query.');
        end if;
    END;
END;
/

/* TRIGGERS */

/* T-1) Integrity Contraints */


/* T-2) Task Languages */
create or replace trigger langcheck
    before insert or update of lang on TaskLanguages
    declare
    isProf boolean;
    cursor c is
    select lang from Profeciencies for update;
    c_rec c%rowtype;
begin
    isProf := false;
    for c_rec in c loop
        if (:new.lang = c_rec.lang) then
            isProf := true;
        end if;
    end loop;
    
    if (isProf = false) then
        raise_application_error(-20006, 'Cannot be program in language not profecient in');
    end if;
end;
/

/* ******************* */
/* (4) Populate Tables */
/* ******************* */

/* Languages */
INSERT INTO LANGUAGES
VALUES ('C++', 'GNU-Test', 'C');

INSERT INTO LANGUAGES
VALUES ('Java', 'ORACLE-Test', 'C');

INSERT INTO LANGUAGES
VALUES ('Python', 'Python-Test', 'I');

INSERT INTO LANGUAGES
VALUES ('Lisp', 'Stanford-Lisp', 'I');

INSERT INTO LANGUAGES
VALUES ('Prolog', 'IBM-Logic-Test', 'C');

/* Tasks */
INSERT INTO Tasks
VALUES ('Zuckerborg', to_date('05-14-1984', 'mm-dd-yyyy'), to_date('08-29-1997', 'mm-dd-yyyy'));

INSERT INTO Tasks
VALUES ('Z Program', to_date('07-23-1973', 'mm-dd-yyyy'), to_date('12-31-2034', 'mm-dd-yyyy'));

INSERT INTO Tasks
VALUES ('FizzBuzz', to_date('01-16-2018', 'mm-dd-yyyy'), to_date('04-30-2018', 'mm-dd-yyyy'));

/* Programmers */
INSERT INTO Programmers
VALUES (000004, 'Makise', 'Kurisu', 35000.00, to_date('07-25-1992', 'mm-dd-yyyy'), 'F', 000004, 'Z Program');

INSERT INTO Programmers
VALUES (000001, 'Okabe', 'Rintaro', 35000.00, to_date('12-14-1991', 'mm-dd-yyyy'), 'M', 000004, 'Z Program');

INSERT INTO Programmers
VALUES (867530, 'Mark', 'Zuckerbuns', 99999999.99, to_date('05-14-1984', 'mm-dd-yyyy'), 'M', 867530, 'Zuckerborg');

INSERT INTO Programmers
VALUES (002049, 'Bill', 'Neye', 123456.78, to_date('11-27-1955', 'mm-dd-yyyy'), 'M', 867530, 'FizzBuzz');

INSERT INTO Programmers
VALUES (001337, 'Shaquille', 'Dontfeel', 1248.16, to_date('11-15-1996', 'mm-dd-yyyy'), 'M', 002049, 'FizzBuzz');

/* Profeciencies */
INSERT INTO Profeciencies
VALUES (000004, 'Lisp', to_date('08-12-2011', 'mm-dd-yyyy'));

INSERT INTO Profeciencies
VALUES (000001, 'Lisp', to_date('08-13-2011', 'mm-dd-yyyy'));

INSERT INTO Profeciencies
VALUES (867530, 'Java', to_date('2-23-1995', 'mm-dd-yyyy'));

INSERT INTO Profeciencies
VALUES (002049, 'C++', to_date('07-12-2013', 'mm-dd-yyyy'));

INSERT INTO Profeciencies
VALUES (001337, 'C++', to_date('06-12-2014', 'mm-dd-yyyy'));

INSERT INTO Profeciencies
VALUES (001337, 'Python', to_date('06-13-2014', 'mm-dd-yyyy'));

/* TaskLanguages */
INSERT INTO TaskLanguages
VALUES ('Zuckerborg', 867530, 'Java');

INSERT INTO TaskLanguages
VALUES ('Z Program', 000004, 'Lisp');

INSERT INTO TaskLanguages
VALUES ('Z Program', 000001, 'Lisp');

INSERT INTO TaskLanguages
VALUES ('FizzBuzz', 002049, 'C++');

INSERT INTO TaskLanguages
VALUES ('FizzBuzz', 001337, 'C++');

INSERT INTO TaskLanguages
VALUES ('FizzBuzz', 001337, 'Python');

/* Completed Tasks */
INSERT INTO CompletedTasks
VALUES ('HaltingProblem');

INSERT INTO CompletedTasks
VALUES ('TravelSalesmen');

/* CompletedTaskProgrammers */
INSERT INTO CompletedTaskProgrammers
VALUES ('leroy', 'jenkins', 'HaltingProblem', 1);

INSERT INTO CompletedTaskProgrammers
VALUES ('Bill', 'Neye', 'HaltingProblem', 356);

INSERT INTO CompletedTaskProgrammers
VALUES ('Mark', 'Zuckerbuns', 'TravelSalesmen', 589);

INSERT INTO CompletedTaskProgrammers
VALUES ('Shaquille', 'Dontfeel', 'TravelSalesmen', 73);

INSERT INTO CompletedTaskProgrammers
VALUES ('Bill', 'Neye', 'TravelSalesmen', 42);

/* CompletedTaskLanguages */
INSERT INTO CompletedTaskLanguages
VALUES ('leroy', 'jenkins', 'HaltingProblem', 'Prolog');

INSERT INTO CompletedTaskLanguages
VALUES ('leroy', 'jenkins', 'HaltingProblem', 'Python');

INSERT INTO CompletedTaskLanguages
VALUES ('Bill', 'Neye', 'HaltingProblem', 'C++');

INSERT INTO CompletedTaskLanguages
VALUES ('Mark', 'Zuckerbuns', 'TravelSalesmen', 'Java');

INSERT INTO CompletedTaskLanguages
VALUES ('Shaquille', 'Dontfeel', 'TravelSalesmen', 'Python');

INSERT INTO CompletedTaskLanguages
VALUES ('Shaquille', 'Dontfeel', 'TravelSalesmen', 'C++');

INSERT INTO CompletedTaskLanguages
VALUES ('Bill', 'Neye', 'TravelSalesmen', 'C++');

/* ******************** */
/* (5) Display Contents */
/* ******************** */

SET LINESIZE 200;
SET WRAP OFF;
SET SERVEROUTPUT ON;

/* Show all tables*/
SELECT table_name
FROM user_tables;

/* Display contents of each table */
SELECT * FROM Tasks;
SELECT * FROM Languages;
SELECT * FROM Programmers;
SELECT * FROM Profeciencies;
SELECT * FROM TaskLanguages;
SELECT * FROM CompletedTasks;
SELECT * FROM CompletedTaskProgrammers;
SELECT * FROM CompletedTaskLanguages;

/* ********************************* */
/* (6) Invoke Queries and Procedures */
/* ********************************* */

/* Q-1) Days remaining */
SELECT name, targetcompletion, trunc(targetcompletion)-trunc(current_date)
AS DaysRemaining
FROM Tasks, Dual
ORDER BY DaysRemaining ASC;

/* Q-2) Tasks Terminating in 2018 */
/* Note to grader, assignment said 2017, but given that this was written a */
/* year ago, 2018 seemed like an acceptable substitute to me. */
SELECT id, fname, lname
FROM Programmers, Tasks
WHERE
    currenttask=Tasks.name
    AND EXTRACT(year FROM Tasks.targetcompletion)=2018;

/* Q-3) Profecient Programmers */
SELECT lang, MIN(profdate)
FROM Profeciencies p RIGHT JOIN Programmers q ON p.programmer=q.id
GROUP BY lang;

/* Q-4) Advisor Pairs */
SELECT p1.fname, p1.lname, p2.fname AS AdvisorF, p2.lname AS AdvistorL
FROM Programmers p1, Programmers p2, Profeciencies q1, Profeciencies q2
WHERE
    p1.id != p1.advisor
    AND p1.advisor = p2.id
    AND q1.programmer = p1.id
    AND q2.programmer = p2.id
    AND q1.lang = q2.lang;
    
/* Invoke procedures */
EXECUTE TerimateTask('FizzBuzz');
-- Python has profecients programmers, so it won't be retired
EXECUTE RetireLanguage('Python');
EXECUTE RetireLanguage('Prolog');
-- Advisees will self advise
EXECUTE RetireProgrammer(1337);
EXECUTE DisplayCompleted('HaltingProblem');

/* ******************* */
/* (7) Update Triggers */
/* ******************* */

/* ********************************* */
/* (8) Remove Components             */
/* ********************************* */
-- Procedures and Triggers

--DROP TRIGGER langcheck;
--DROP PROCEDURE TerimateTask;
--DROP PROCEDURE RetireLanguage;
--DROP PROCEDURE RetireProgrammer;
--DROP PROCEDURE DisplayCompleted;

-- Tables

--DROP TABLE Profeciencies;
--DROP TABLE TaskLanguages;
--DROP TABLE CompletedTaskLanguages;
--DROP TABLE Languages;
--DROP TABLE CompletedTaskProgrammers;
--DROP TABLE CompletedTasks;
--DROP TABLE Programmers;
--DROP TABLE Tasks;
