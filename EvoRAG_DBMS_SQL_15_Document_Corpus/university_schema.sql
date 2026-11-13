-- EvoRAG DBMS corpus: optional teaching schema for PostgreSQL 18
-- Run only in a disposable practice database.
CREATE TABLE departments (
    dept_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dept_name TEXT NOT NULL UNIQUE
);

CREATE TABLE students (
    student_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    dept_id BIGINT REFERENCES departments(dept_id),
    enrollment_year SMALLINT NOT NULL CHECK (enrollment_year BETWEEN 2000 AND 2100)
);

CREATE TABLE instructors (
    instructor_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    dept_id BIGINT NOT NULL REFERENCES departments(dept_id)
);

CREATE TABLE courses (
    course_id TEXT PRIMARY KEY,
    course_title TEXT NOT NULL,
    credits SMALLINT NOT NULL CHECK (credits BETWEEN 1 AND 6),
    dept_id BIGINT NOT NULL REFERENCES departments(dept_id),
    instructor_id BIGINT REFERENCES instructors(instructor_id)
);

CREATE TABLE enrollments (
    student_id BIGINT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    course_id TEXT NOT NULL REFERENCES courses(course_id) ON DELETE RESTRICT,
    semester TEXT NOT NULL,
    grade NUMERIC(5,2) CHECK (grade BETWEEN 0 AND 100),
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (student_id, course_id, semester)
);

CREATE INDEX idx_students_department ON students(dept_id);
CREATE INDEX idx_courses_department ON courses(dept_id);
CREATE INDEX idx_enrollments_course ON enrollments(course_id);
CREATE INDEX idx_enrollments_student ON enrollments(student_id);

-- Example seed data. Names/emails are synthetic.
INSERT INTO departments (dept_name) VALUES
    ('Computer Science'), ('Mathematics'), ('Electronics');

INSERT INTO students (full_name, email, dept_id, enrollment_year)
SELECT 'Aarav Mehta', 'aarav@example.edu', dept_id, 2025
FROM departments WHERE dept_name = 'Computer Science';

INSERT INTO instructors (full_name, email, dept_id)
SELECT 'Dr. Nisha Rao', 'nisha.rao@example.edu', dept_id
FROM departments WHERE dept_name = 'Computer Science';

INSERT INTO courses (course_id, course_title, credits, dept_id, instructor_id)
SELECT 'CS301', 'Database Management Systems', 4, d.dept_id, i.instructor_id
FROM departments AS d CROSS JOIN instructors AS i
WHERE d.dept_name = 'Computer Science'
  AND i.email = 'nisha.rao@example.edu';

INSERT INTO enrollments (student_id, course_id, semester, grade)
SELECT s.student_id, c.course_id, '2026-Fall', 91.50
FROM students AS s CROSS JOIN courses AS c
WHERE s.email = 'aarav@example.edu' AND c.course_id = 'CS301';
