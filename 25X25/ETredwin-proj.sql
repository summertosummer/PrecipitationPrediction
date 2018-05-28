create table Task(
	Name char(15) PRIMARY KEY,
	Start_date date,
	Target_date date
);


create table Programmer(
	ID Number(6) PRIMARY KEY,
	Name Char(15),
	Gender Char(15),
 	Salary Number(6),
	DOB Date,
	Curently_Working char(1)
);

create table Programming_Language(
	Name Char(15) PRIMARY KEY,
	Comp_Interp Char(15),
	Examination Char(15)
);


create table Works_on(
	TaskName Char(15),
	ProgrammerID Number(6),
	From_date Date,
	PRIMARY KEY (TaskName, ProgrammerID),
	FOREIGN KEY (TaskName) REFERENCES Task(Name),
	FOREIGN KEY (ProgrammerID) REFERENCES Programmer(ID)
);

create table Used_in(
	TaskName Char(15),
	Programming_Lang Char(15),
	PRIMARY KEY (TaskName, Programming_Lang),
	FOREIGN KEY (TaskName) REFERENCES Task(Name),
	FOREIGN KEY (Programming_Lang) REFERENCES Programming_Language(Name)
);

create table Proficient_in(
	ProgrammerID Number(6),
	Programming_Lang Char(15),
	Date_of Date,
	PRIMARY KEY (ProgrammerID, Programming_Lang),
	FOREIGN KEY (ProgrammerID) REFERENCES Programmer(ID),
	FOREIGN KEY (Programming_Lang) REFERENCES Programming_Language(Name)
);

create table Supervises(
	SupervisorID Number(6),
	ProgrammerID Number(6) PRIMARY KEY,
	FOREIGN KEY (ProgrammerID) REFERENCES Programmer(ID),
	FOREIGN KEY (SupervisorID) REFERENCES Programmer(ID)
);

create table Terminated(
	TaskName Char(15) PRIMARY KEY,
	Termination_date Date,
	FOREIGN KEY (TaskName) REFERENCES Task(Name)
);


INSERT INTO Programmer VALUES( '1', 'Yash', 'Male', '1000', To_date('03-29-2017','mm-dd-yyyy'), 'y');
INSERT INTO Programmer VALUES( '2', 'Tim', 'Car', '2000', To_date('03-23-2017','mm-dd-yyyy'), 'y');
INSERT INTO Programmer VALUES( '3', 'Mason', 'Male', '1000',To_date('03-24-2017','mm-dd-yyyy'), 'y');
INSERT INTO Programmer VALUES( '4', 'Flocky', 'Fluid', '2000', To_date('03-03-2017','mm-dd-yyyy'), 'n');

INSERT INTO Supervises VALUES('1', '1');
INSERT INTO Supervises VALUES('2', '2');
INSERT INTO Supervises VALUES('2', '3');
INSERT INTO Supervises VALUES('1', '4');

INSERT INTO Programming_Language VALUES('C', 'Compiled', 'Exam_C');
INSERT INTO Programming_Language VALUES('JAVA', 'Compiled', 'Exam_JAVA');
INSERT INTO Programming_Language VALUES('LISP', 'Interperated', 'Exam_LISP');
INSERT INTO Programming_Language VALUES('ALGOL', 'Compiled', 'Exam_ALGOL');
INSERT INTO Programming_Language VALUES('FORTRAN', 'Compiled', 'Exam_FORTRAN');

INSERT INTO Task VALUES('GAMES', To_date('3-27-2018','mm-dd-yyyy'), To_date('5-24-2018','mm-dd-yyyy'));
INSERT INTO Task VALUES('WORD', To_date('2-25-2017','mm-dd-yyyy'), To_date('7-27-2017','mm-dd-yyyy'));
INSERT INTO Task VALUES('SORT', To_date('6-26-2017','mm-dd-yyyy'), To_date('8-28-2018','mm-dd-yyyy'));

INSERT INTO Works_on VALUES('GAMES', '1', To_date('4-20-2017','mm-dd-yyyy'));
INSERT INTO Works_on VALUES('GAMES', '2', To_date('7-20-2017','mm-dd-yyyy'));
INSERT INTO Works_on VALUES('WORD', '3', To_date('7-20-2017','mm-dd-yyyy'));
INSERT INTO Works_on VALUES('SORT', '4', To_date('7-20-2017','mm-dd-yyyy'));

INSERT INTO Proficient_in VALUES('1', 'C', To_date('4-20-2017','mm-dd-yyyy'));
INSERT INTO Proficient_in VALUES('1', 'JAVA', To_date('5-22-2017','mm-dd-yyyy'));
INSERT INTO Proficient_in VALUES('1', 'LISP', To_date('2-27-2017','mm-dd-yyyy'));
INSERT INTO Proficient_in VALUES('1', 'ALGOL', To_date('5-5-2016','mm-dd-yyyy'));
INSERT INTO Proficient_in VALUES('2', 'FORTRAN', To_date('7-29-1998','mm-dd-yyyy'));
INSERT INTO Proficient_in VALUES('3', 'C', To_date('5-14-1997','mm-dd-yyyy'));
INSERT INTO Proficient_in VALUES('4', 'LISP',To_date('5-5-1889','mm-dd-yyyy'));

INSERT INTO Used_in VALUES('GAMES', 'JAVA');
INSERT INTO Used_in VALUES('GAMES', 'C');
INSERT INTO Used_in VALUES('GAMES', 'FORTRAN');
INSERT INTO Used_in VALUES('WORD', 'C');
INSERT INTO Used_in VALUES('SORT', 'LISP');

CREATE OR REPLACE PROCEDURE Terminate_task(NAME IN Task.Name%type) AS
        begin
                INSERT INTO Terminated VALUES( NAME , To_date(SYSDATE, 'dd-mm-yyyy'));
        end;
/
/*
CREATE OR REPLACE PROCEDURE Retire_lang(Name IN Programming_Language.Name%type)

begin
        IF Name IN SELECT Used_in.Programming_Lang FROM Used_in
        THEN
        ELSEIF NAME IN SELECT
        THEN
        ELSE
end;
*/
CREATE OR REPLACE PROCEDURE Terminated_task(Terminated_Name IN Terminated.Name%type) AS
BEGIN
        IF EXISTS (SELECT Terminated.TaskName FROM Terminated WHERE Terminated.TaskName = Terminated_Name)
        THEN
                SELECT Programmer.Name, (Terminated.Termination_date - Works_on.Start_date) "Days"
                FROM Programmer, Works_on, Terminated
                WHERE Terminated.TaskName = Terminated_Name
                AND Works_on.Name = Terminated_Name
                AND Works_on.ProgrammerID = Programmer.ID;
        else
                DBMS_OUTPUT.PUT_LINE('Terminated Task Does Not Exist');
        end if;

END;
/
show errors;
/*
CREATE OR REPLACE TRIGGER programmer_works_proficiency
        AFTER INSERT ON Works_on
BEGIN
        FOR ROW IN (SELECT * FROM (Used_in JOIN Works_on ON (Used_in.TaskName) INNER JOIN Proficient_in ON Proficient_in.ProgrammerID )
                IF ROW
*/



--Q1
SELECT
	Task.Name, Task.Target_date, MONTHS_BETWEEN(Task.Target_date, to_date(SYSDATE, 'dd-mm-yyyy') ) - 24000 "Months"
FROM
	Task
WHERE
	(MONTHS_BETWEEN(Task.Target_date, to_date(SYSDATE, 'dd-mm-yyyy') ) - 24000) > 0;

--Q2

SELECT
	Programmer.ID, Programmer.Name
FROM
	Programmer, Works_on, Task
WHERE
	Programmer.ID = Works_on.ProgrammerID
	AND
	Task.Target_date < to_date('31-12-2018', 'dd-mm-yyyy')
	AND
	Task.Target_date > to_date('31-12-2017', 'dd-mm-yyyy')
	AND
	Task.Name = Works_on.TaskName;


--Q3
SELECT Sel.Programming_Lang, Programmer.Name, Sel.PROFICIENT_FROM
FROM
	(SELECT Proficient_in.Programming_Lang, MIN(Proficient_in.Date_of) "PROFICIENT_FROM"
	FROM Used_in JOIN Proficient_in ON Used_in.Programming_Lang = Proficient_in.Programming_Lang
	GROUP BY  Proficient_in.Programming_Lang) "SEL" JOIN Proficient_in ON Proficient_in.Date_of = SEL.PROFICIENT_FROM JOIN Programmer ON Proficient_in.ProgrammerID = Programmer.ID;




--Q4
SELECT Supervises.ProgrammerID, P2.NAME, Supervises.SupervisorID, P1.Name, T1.Programming_Lang
FROM Programmer P1, Programmer P2, Proficient_in T1, Proficient_in T2, Supervises
WHERE T1.Programming_Lang = T2.Programming_Lang
AND T1.ProgrammerID = Supervises.ProgrammerID
AND T2.ProgrammerID = Supervises.SupervisorID
AND P1.ID = T1.ProgrammerID
AND P2.ID = T2.ProgrammerID
AND P1.ID != P2.ID;














/*
DROP TABLE Task CASCADE constraints;
DROP TABLE Programmer CASCADE constraints;
DROP TABLE Programming_Language CASCADE constraints;
DROP TABLE Works_on CASCADE constraints;
DROP TABLE Used_in CASCADE constraints;
DROP TABLE Proficient_in CASCADE constraints;
DROP TABLE Supervises CASCADE constraints;
DROP TABLE Terminated CASCADE constraints;
*/
