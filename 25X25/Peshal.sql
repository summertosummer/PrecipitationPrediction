create table task (
name        VARCHAR2(15),
start_date  DATE,
target_date DATE,
constraint task_pk primary key (name)
)
;
------------------------------------------
create table completed_task (
name             VARCHAR2(15),
completion_date  DATE,
status           VARCHAR2(10),
constraint completed_task_pk primary key (name)
)
;
------------------------------------------
create table programmer (
id            number(6) NOT NULL,
name          VARCHAR2(15),  
gender        char(1),
yearly_salary number(9,2), 
job_title     VARCHAR2(10),  
mgr_id        number(6),
date_of_birth DATE, 
constraint programmer_pk primary key (id)
)
;
----------------------------------------
create table inactive_programmer (
id             number(6) NOT NULL,
resigning_date DATE,
constraint active_programmer_pk primary key (id)
)
;
----------------------------------------
create table language (
name          VARCHAR2(15),
exam_suite    VARCHAR2(15),    
type          VARCHAR2(15),  
constraint language_pk primary key (name)
)
;
--------------------------------------------
create table assignedTask (
task_name     VARCHAR2(15),
programmer_id number(6),
assigned_date DATE, 
constraint assignedTask_pk primary key (task_name, programmer_id)
)
;

-----------------------------------------------------------
create table  proficiency(
language_name       VARCHAR2(15),
programmer_id       number(6),
date_of_proficiency DATE, 
constraint profiency_pk primary key (language_name, programmer_id)
)
;

-----------------------------------------------
create table  usage(
language_name       VARCHAR2(15),
programmer_id       number(6),
task_name           VARCHAR2(15),
date_of_proficiency DATE, 
constraint usage_pk primary key (programmer_id, task_name)
)
;

--------------------------------------------------
create table  completedBy(
completed_task_name VARCHAR2(15),
programmer_id       number(6),
no_of_days          number(6),
constraint completedBy_pk primary key (programmer_id, completed_task_name)
)
;


----------------------------------------Inserting the values in the table----------------------------
-----------Populate Tasks-----------
insert into task values ( 'Canyon', TO_DATE('2003/05/03', 'yyyy/mm/dd'),TO_DATE('2013/05/03', 'yyyy/mm/dd'));
insert into task values ( 'Blueberry', SYSDATE,SYSDATE);
insert into task values ( 'New Task', TO_DATE('2013/05/03', 'yyyy/mm/dd'),TO_DATE('2018/05/03', 'yyyy/mm/dd'));


-----------Populate CompletedTask-----------
insert into completed_task values ( 'Canyon', SYSDATE,'COMPLETED');

-----------Populate Programmers-----------
-- Increase varchar for position/title
insert into programmer values ( 111, 'Jones', 'M', 60000, 'Engineer', 222, TO_DATE('1990/05/03', 'yyyy/mm/dd'));
insert into programmer values ( 122, 'Jane', 'F', 50000, 'Designer', 222, TO_DATE('1988/05/03', 'yyyy/mm/dd'));
insert into programmer values ( 123, 'Alex', 'M', 60000, 'QA', 222, TO_DATE('1991/05/03', 'yyyy/mm/dd'));
insert into programmer values ( 222, 'Lex', 'M', 60000, 'Manager', NULL, TO_DATE('1990/05/03', 'yyyy/mm/dd'));

----------Inactive Programmer--------------------
insert into inactive_programmer values ( 111, SYSDATE);

----------Language--------------------
-- Increase var char for certification
insert into language values ( 'java', 'Oracle', 'compiled');
insert into language values ( 'python', 'Pythontest', 'interpreter');

----------Assigned Task--------------------
insert into assignedTask values ( 'Canyon', 111,  TO_DATE('2016/05/03', 'yyyy/mm/dd'));
insert into assignedTask values ( 'Canyon', 122,  TO_DATE('2015/05/03', 'yyyy/mm/dd'));
insert into assignedTask values ( 'Blueberry', 123, TO_DATE('2016/05/03', 'yyyy/mm/dd'));

----------Profieciency--------------------
insert into proficiency values ( 'java', 111, TO_DATE('2015/05/03', 'yyyy/mm/dd'));
insert into proficiency values ( 'python', 122, TO_DATE('2016/05/03', 'yyyy/mm/dd'));
insert into proficiency values ( 'python', 123, TO_DATE('2016/04/03', 'yyyy/mm/dd'));
insert into proficiency values ( 'java', 123, TO_DATE('2016/05/03', 'yyyy/mm/dd'));
----------USAGE--------------------
insert into usage values ( 'java', 111, 'Canyon');
insert into usage values ( 'python', 123, 'Blueberry');

----------completedBy--------------------
insert into completedBy values ( 'Canyon', 111, 45);
insert into completedBy values ( 'Canyon', 122, 120);


--select 'drop table ', table_name, 'cascade constraints;' from user_tables;


-------------------------------------Select Statements------------------------------------------------

select  name, start_date, target_date, TRUNC(target_date) - TRUNC(SYSDATE) as "REM", round(months_between(target_date,SYSDATE))  as "Mo Days" from task where target_date > SYSDATE;

--2-------------
select id, name from programmer where id = (select programmer_id from assignedTask where assignedtask.task_name = (select name from task where (EXTRACT(YEAR FROM target_date) = '2017')));

--3----------------------------
select language_name , programmer_id from proficiency d where date_of_proficiency = ( SELECT MIN(date_of_proficiency) FROM proficiency  e GROUP BY language_name having e.language_name = d.language_name);
--4------------------------------
select e1.id, e1.name , e2.programmer_id from programmer e1,  proficiency e2 where (e1.id = e2.programmer_id and e1.mgr_id = e2.programmer_id and e1.mgr_id != e1.id);

--P1----------------
insert into completed_task(values name, SYSDATE, "TERMINATED");

--P2-----------------------------


--P3-------------------------
update table programmer set status="LEFT" where name = "";

--P4----------------------------
select e.id, e.name from programmer e , assignedtask p where (e.id = p.programmer_id and p.task_name='Canyon');


--E1-----------------
select e.id, e.name from programmer e, assignedtask p where (e.id = p.programmer_id);

--E2-------------------------------
SELECT task_name, COUNT(task_name)   
FROM assignedTask  GROUP BY task_name   
HAVING COUNT (task_name)=(

SELECT MAX (mycount) FROM (SELECT task_name, COUNT(task_name) mycount  FROM assignedTask  GROUP BY task_name));  




select e.programmer_id from proficiency e, proficiency d where (e.programmer_id = d.programmer_id and e.language_name = 'java');

select e.programmer_id , count(e.language_name) from proficiency e , proficiency e1 where
(e.programmer_id = e1.programmer_id and e.language_name = e1.language_name and e.language_name='java')
select e.programmer_id , e1.programmer_id from proficiency e , proficiency e1 where (e.programmer_id = e1.programmer_id and e.language_name = e1.language_name and e.language_name='java')


select programmer_id, count(language_name) from proficiency group by programmer_id having 
(count(language_name) = 1);

select programmer_id, count(language_name) from proficiency 
-- select programmer_id from proficiency where(count(language_name) = 1);
-- select programmer_id , language_name from proficiency where (language_name = 'java');



-------------------------------Procedure--------------------------------------------------
---------------------------------------------------------------------

create or replace procedure give_task
(task_name IN task.name%type)
is
begin
select * from task where name = task_name
end;
/




create or replace procedure get_sal(
id1 IN emp.id%type,
sal1 OUT emp.salary%type)
is
begin
select salary INTO sal1
from emp
where id = id1;
end;


create or replace procedure get_my_task(
tname IN task.name%type,
task1 OUT task.name%type)
is
begin
select name INTO task1 from task where name = tname;
end;

variable s varchar2(15)



-----------------------------------------------Drop Table-----------------------

drop table COMpleted_task;
drop table language;
drop table programmer;
drop table task;
