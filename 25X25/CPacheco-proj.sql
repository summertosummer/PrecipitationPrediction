/*------------------------------*/
-- CSE373 Homework 7            -- 
-- Celia M. Pacheco             -- 
-- April 30 2018		--
-- CODE FILE                    --
-- The project creates several  --
-- tables used in for a company --
-- and implements different     --
-- queries and triggers         --
/*------------------------------*/

set echo off;
set serveroutput on;
set linesize 200;
/*-------TASK------*/ 
drop table task cascade constraints
;

create table task ( 
  task_name varchar2(15),
  start_date date,
  target_date date,
  constraint task_pk primary key (task_name)
)
;

/*-------COMPETED TASKS-------*/
drop table comp_task cascade constraint
;
create table comp_task (
  ctask_name varchar2(15),
  completed_date date,
  constraint comp_task_pk primary key (ctask_name),
  constraint comp_task_fk foreign key (ctask_name) references task(task_name)
)
; 

/*-------PROGRAMMER-------*/
drop table programmer cascade constraint
;
create table programmer (
  ID number(6),
  first_name varchar2(15),
  last_name varchar2(15),
  dob date, 
  job_title varchar2(15), 
  gender char(1),
  yearly_salary number(11,2),
  supervisor_ID number(6),
  employment_status varchar2(4), -- used to keep track of employment status
  constraint programmer_pk primary key (ID),
  constraint programmer_sal_pos check (yearly_salary >= 0)
)
;

/*-------LANGUAGE-------*/
drop table language cascade constraint
;
create table language (
  name varchar2(15),
  l_type varchar2(15),
  exam_suite varchar2(15),
  constraint language_pk primary key (name)
)
;

/*-------ASSIGNMENT-------*/
drop table assignment cascade constraint
;
create table assignment (
  task_name varchar2(15),
  PID number(6),
  language varchar2(15), --used to keep track of langauge being used in task
  hours_worked number(3),
  constraint assignment_pk primary key(task_name, PID),
  constraint assignment_task_fk foreign key (task_name) references task(task_name),
  constraint assignment_PID_fk foreign key (PID) references programmer(ID),
  constraint assignment_language_fk foreign key (language) references language(name)
)
;

/*-------PROFICIENCY-------*/
drop table proficiency cascade constraint
;
create table proficiency (
 language varchar2(15), 
 PID number(6),
 exam_date date,
 constraint proficiency_pk primary key(language, PID),
 constraint proficiency_language_fk foreign key (language) references language(name),
 constraint proficiency_PID_fk foreign key (PID) references programmer(ID)
)
;  

/*--terminate a task --*/
CREATE OR REPLACE PROCEDURE TerminateTask 
 (task in task.task_name%type) 
IS 
BEGIN
--just insert task into completed tasks table
 INSERT INTO comp_task VALUES (task, sysdate);
 commit;
END; 
/

/*-- retire a language--*/
CREATE OR REPLACE PROCEDURE RetireLang
 (lang in language.name%type)
IS
-- cursor used to check if language is used in current task
 cursor c1 is
 select language
 from assignment
 where language = lang
 and task_name not in
  (select ctask_name from comp_task);

 c1_rec c1%rowtype;

-- cursor used to check if language is proficienty by anyone 
 cursor c2 is
 select language, pid
 from proficiency
 where language = lang;

 c2_rec c2%rowtype;

-- used to store number of proficiencies for programmer
 count_prof number;
 
BEGIN
--open cursors
  if not c1%isopen then
   open c1;
  end if;
  fetch c1 into c1_rec;

  if not c2%isopen then
   open c2;
  end if;
  fetch c2 into c2_rec;

--if c1 not found then language isn't used in current task 
  if c1%notfound then
-- if c2 is found then a programmer is proficient in it
   if c2%found then
-- check howmany proficiencies the programmer has
    select count(*) into count_prof 
    from proficiency
    where pid = c2_rec.pid
    group by pid;
-- if proficiencies are > 1 then programmer will not be left 0 proficiencies
     if count_prof > 1 then
      DELETE from proficiency
      where language = lang;
     else
      dbms_output.put_line('cannot delete language, leaves programmer with 0 proficiencies'); 
     end if;
   end if;
   DELETE FROM language
   WHERE name = lang;
  else
   dbms_output.put_line('cannot delete language is being used in an open task');
  end if;
 commit;
 close c1;
 close c2;
END;
/ 

/*-------Update employment status of employee leaving-------*/
CREATE OR REPLACE PROCEDURE LeavingP
 (emp in programmer.id%type)
IS
 cursor c1 is
 select id
 from programmer
 where id = emp;

 c1_rec c1%rowtype;

BEGIN
 if not c1%isopen then
  open c1;
 end if;
 fetch c1 into c1_rec;

 if c1%found then
  update programmer
  set employment_status = 'term'
  where id = emp;
 end if;

 commit;
 close c1;
END;
/

/*--get name and hours worked from employess who have worked on a specific terminated task-- */
CREATE OR REPLACE PROCEDURE TermTaskInfo
 (name in comp_task.ctask_name%type)
IS
 cursor c1 is
 select first_name, last_name, assignment.hours_worked as hours 
 from programmer, comp_task, assignment 
 where comp_task.ctask_name = name
 and comp_task.ctask_name = assignment.task_name
 and programmer.id = assignment.pid;

 c1_rec c1%rowtype;

 BEGIN
 if not c1%isopen then
  open c1;
 end if;
 fetch c1 into c1_rec;
 
 while c1%found loop
  dbms_output.put_line(c1_rec.first_name || ' ' || c1_rec.last_name || ' ' || c1_rec.hours);
  fetch c1 into c1_rec;
 end loop;
 close c1;
END;
/  

/*--TRIGGER to check if prorammer is assigned to task using a language 
they are proficient in
*/
CREATE OR REPLACE TRIGGER assign_proficiency
 before insert on assignment
 for each row
DECLARE
 lan number;
BEGIN
  select count(*) 
  into lan
  from proficiency
  where pid = :new.pid
  and language = :new.language;

  if lan = 0  
  then
      raise_application_error(-20000, 'Cannot assign task to programmer in language they are not proficient');
  end if;
END;  
/

/*-- TRIGGER checks and on programmer is assigned to only one open task --*/
CREATE OR REPLACE TRIGGER chk_multiple_assign
 before insert on assignment
 for each row
DECLARE
 lan number;
BEGIN
 select count(*)
 into lan
 from assignment
 where pid = :new.pid
 and task_name not in
  (select ctask_name from comp_task);

 if lan > 0
 then 
  raise_application_error(-20001, 'Cannot assign programmer to multiple open tasks');
 end if;
END;
/
show errors
/*-------POPULATE TABLES-------*/
insert into task values ('foo',TO_DATE('01-jan-2018', 'DD-MON-YYYY'), TO_DATE('01-feb-2018','DD-MON-YYYY'));
insert into task values ('bar', TO_DATE('01-dec-2017','DD-MON-YYYY'), TO_DATE('15-may-2018','DD-MON-YYYY'));
insert into task values ('fun', TO_DATE('05-apr-2018', 'DD-MON-YYYY'), TO_DATE('20-jun-2019', 'DD-MON-YYYY'));
insert into task values ('bus', TO_DATE('16-mar-2018', 'DD-MON-YYYY'), TO_DATE('22-may-2018', 'DD-MON-YYYY'));
insert into comp_task values('foo', TO_DATE('02-feb-2018', 'DD-MON-YYYY'));
insert into programmer values (14, 'Fred', 'Jones',TO_DATE('16-mar-1980','DD-MON-YYYY'), 'supervisor', 'M', 120000.00, 14, 'emp');
insert into programmer values (1, 'Velma', 'Dinkley', TO_DATE('02-jun-1971','DD-MON-YYYY') , 'supervisor', 'F', 500000.00, 1, 'emp');
insert into programmer values(5, 'Scooby', 'Doo', TO_DATE('03-aug-1989', 'DD-MON-YYYY'), 'programmer', 'M', 250000, 1, 'emp');
insert into programmer values(3, 'Shaggy', 'Rogers', TO_DATE('11-nov-1992', 'DD-MON-YYYY'), 'programmer', 'M', 60000, 14, 'emp');
insert into language values ('python', 'interpreted', 'superpythonexam');
insert into language values ('C', 'compiled', 'theCexam');
insert into language values ('ruby', 'interpreted', 'RubyExam');
insert into language values ('basic', 'compiled', 'basicExam');
insert into proficiency values ('python', 14, TO_DATE('20-nov-2008','DD-MON-YYYY'));
insert into proficiency values ('C', 5, TO_DATE('08-sep-2000', 'DD-MON-YYYY'));
insert into proficiency values ('ruby', 1, TO_DATE('18-may-1997'));
insert into proficiency values ('ruby', 5, TO_DATE('22-feb-2016'));
insert into proficiency values ('python', 3, TO_DATE('17-oct-2011'));
insert into proficiency values ('basic', 1, TO_DATE('22-jun-2011'));
insert into assignment values ('foo', 14, 'python', 60);
insert into assignment values ('bar', 1, 'ruby', 18);
insert into assignment values ('fun', 3, 'python', 2);

select * from task;
select * from comp_task;
select * from programmer;
select * from language;
select * from assignment;


/*-------QUERY1-------*/
select task_name, target_date, months_between(target_date, sysdate) as "months remaining" 
from task 
where task_name not in 
 (select ctask_name from comp_task) 
order by "months remaining" asc 
; 

/*-------QUERY2-------*/
select ID, first_name, last_name 
from programmer 
where ID in 
 (select PID 
  from assignment 
  where task_name in 
  (select task_name 
   from task 
   where target_date between TO_DATE('01-jan-2018', 'DD-MON-YYYY') and TO_DATE('31-dec-2018', 'DD-MON-YYYY')
  )
 ) 
order by ID asc 
; 

/*-------QUERY3-------*/
select pid, language, exam_date 
from proficiency p1 
where exam_date = 
 (select min(exam_date)
  from proficiency p2
  where p2.language = p1.language
 )
and
language in
 (select language
  from assignment
  where task_name not in
  (select ctask_name
   from comp_task
  )
 )
order by pid asc
; 

/*-------QUERY4-------*/
select pp1.language, p1.id as "programmer id", p1.first_name as "programmer first name", p2.id as "supervisor id", p2.first_name as "supervisor first name"
from programmer p1, programmer p2, proficiency pp1, proficiency pp2
where p1.id <> p1.supervisor_ID
and p1.supervisor_ID = p2.ID
and pp1.pid = p1.id
and pp2.pid = p2.id
and pp2.language = pp1.language
order by pp1.language asc
;


execute TerminateTask('bar');
select * from comp_task;

execute RetireLang('C');
execute RetireLang('basic');
select * from language;

execute LeavingP(3);
select * from programmer;

execute TermTaskInfo('bar');

insert into assignment values ('fun', 5, 'python', 18);
insert into assignment values ('foo', 3, 'python', 9);


-- Here is the codeblock that should do the clean up
/*
drop trigger assign_proficiency;
drop trigger chk_multiple_assign;

drop table task cascade constraints;
drop table comp_task cascade constraints;
drop table programmer cascade constraints;
drop table assignment cascade constraints;
drop table proficiency cascade constraints; 
drop table language cascade constraints;

drop procedure TerminateTask;
drop procedure RetireLang;
drop procedure LeavingP;
drop procedure TermTaskInfo;
*/

