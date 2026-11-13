# EvoRAG Task 9A - Answer Key for Frozen Prediction Run

**Run:** `20261109_151957Z`  
**Frozen outputs:** 32 questions x 3 systems = 96 predictions  
**Blind-run metadata:** `answer_key_loaded=false`; the answer key was not loaded during inference.

Use this key only after freezing predictions. Accept semantically equivalent correct answers; do not require exact wording. For SQL, validate syntax and result semantics against the supplied PostgreSQL 18 university schema before awarding full correctness.

## DBMS-EVAL-0041 - direct_factual (easy)

**Question:** What is the purpose of a buffer manager?

**Answerable from corpus:** Yes

**Expected answer:**

The buffer manager manages an in-memory cache of database pages, allowing the DBMS to reuse pages and reduce repeated storage I/O.

**Supporting sources:**

- `08_database_storage_pages_and_buffer_management.pdf` (source `DBMS-08`, PDF page 3)

**Evidence excerpts:**

- `08_database_storage_pages_and_buffer_management.pdf` p. 3: retrieval corpus - original explanations with cited references Page 3 3. Buffer pool and page lifecycle The buffer pool is a memory area managed by the DBMS to cache pages read from persistent storage. A page request first checks whether the page is already resident. If it is, the DBMS can avoid a storage
  - Benchmark evidence-anchor SHA-256: `3df054e50ba1f2c1d30540c1ca931d2b8980a9d405d4761a948ffa7d7f72b58e`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0042 - direct_factual (easy)

**Question:** What is a database index?

**Answerable from corpus:** Yes

**Expected answer:**

An index is an auxiliary data structure that helps the DBMS locate rows without scanning every page of a table. It stores search-key information and a way to find corresponding rows or pages.

**Supporting sources:**

- `09_index_structures_btrees_hashing_and_postgresql.pdf` (source `DBMS-09`, PDF page 1)

**Evidence excerpts:**

- `09_index_structures_btrees_hashing_and_postgresql.pdf` p. 1: and range access DBMS & SQL Knowledge Base | EvoRAG corpus | PostgreSQL 18 baseline 1. Why indexes exist An index is an auxiliary data structure that helps a DBMS locate rows without scanning every page of a table. An index usually stores search-key values with a way to find the corresponding row or data page. It
  - Benchmark evidence-anchor SHA-256: `fe07eec534c27e7818ffdb514d3ec1a6c9d2bea3dddda94e78584206f59e7895`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0043 - direct_factual (easy)

**Question:** Which index type is generally appropriate for equality and range lookups in PostgreSQL?

**Answerable from corpus:** Yes

**Expected answer:**

A B-tree index is the general-purpose default in PostgreSQL and supports equality and ordered/range comparisons.

**Supporting sources:**

- `09_index_structures_btrees_hashing_and_postgresql.pdf` (source `DBMS-09`, PDF page 3)

**Evidence excerpts:**

- `09_index_structures_btrees_hashing_and_postgresql.pdf` p. 3: handling and bucket growth are central design problems. PostgreSQL supports multiple index access methods: • B-tree: general-purpose equality, range, and ordered access. • Hash: equality comparisons. • GiST: extensible framework for a range of search strategies, including geometric and range-like operations. • SP-GiST: partitioned search
  - Benchmark evidence-anchor SHA-256: `b7b179e2005f982242c555bec6b5045ce3addd76d1e7ca06c19c641f14326264`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0044 - direct_factual (easy)

**Question:** What is the main use case of a hash index?

**Answerable from corpus:** Yes

**Expected answer:**

A hash index is aimed at equality comparisons; it does not provide the ordered traversal needed for ordinary range scans.

**Supporting sources:**

- `09_index_structures_btrees_hashing_and_postgresql.pdf` (source `DBMS-09`, PDF page 3)

**Evidence excerpts:**

- `09_index_structures_btrees_hashing_and_postgresql.pdf` p. 3: supports multiple index access methods: • B-tree: general-purpose equality, range, and ordered access. • Hash: equality comparisons. • GiST: extensible framework for a range of search strategies, including geometric and range-like operations. • SP-GiST: partitioned search structures for data that can
  - Benchmark evidence-anchor SHA-256: `8816bcae544538a8be840f79e4d2231e529fda8c526c1955dfe7df952b260693`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0045 - direct_factual (easy)

**Question:** What does atomicity mean in ACID?

**Answerable from corpus:** Yes

**Expected answer:**

Atomicity means all effects of a transaction are applied or none are. A failed transaction must not leave a partially committed business operation.

**Supporting sources:**

- `10_transactions_and_acid_properties.pdf` (source `DBMS-10`, PDF page 2)

**Evidence excerpts:**

- `10_transactions_and_acid_properties.pdf` p. 2: Educational retrieval corpus - original explanations with cited references Page 2 2. ACID properties Atomicity: all effects of a transaction are applied or none are. If a statement fails, the transaction may need rollback; applications should handle errors and transaction state correctly. Consistency: a transaction
  - Benchmark evidence-anchor SHA-256: `41525800f77728769f2888a95c14e20cca99b4720f32177458c36f8b9e829a20`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0046 - direct_factual (easy)

**Question:** What does durability mean in ACID?

**Answerable from corpus:** Yes

**Expected answer:**

Durability means that once a transaction commits, its committed effects survive the failures covered by the DBMS durability guarantee.

**Supporting sources:**

- `10_transactions_and_acid_properties.pdf` (source `DBMS-10`, PDF page 2)

**Evidence excerpts:**

- `10_transactions_and_acid_properties.pdf` p. 2: concurrency, and performance. The exact phenomena permitted depend on the DBMS and isolation implementation. Durability: once a commit is acknowledged under the configured durability guarantees, the committed result should survive specified failures. Logging, flush policy, storage guarantees, replication, and configuration
  - Benchmark evidence-anchor SHA-256: `d65259bf995b40280fd91a83ffa3de1c79e0e3c1fa14310ab807f48b39faf88f`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0047 - direct_factual (medium)

**Question:** What is conflict serializability?

**Answerable from corpus:** Yes

**Expected answer:**

A schedule is conflict-serializable if it is conflict-equivalent to some serial schedule. A precedence graph with no cycle is a standard test for conflict serializability.

**Supporting sources:**

- `11_concurrency_control_serializability_mvcc.pdf` (source `DBMS-11`, PDF page 2)

**Evidence excerpts:**

- `11_concurrency_control_serializability_mvcc.pdf` p. 2: T_i -> T_j when an operation in T_i occurs before and conflicts with an operation in T_j on the same item. A schedule is conflict-serializable if the graph is acyclic. A topological ordering of the graph gives an equivalent serial order. Example intuition If T1 writes account A before T2 reads A, the schedule requires
  - Benchmark evidence-anchor SHA-256: `b3bfd2fb209d86be9c4fd08fdf93d0b1e6be4eb4e90b5b1059d01c849a7a964f`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0048 - direct_factual (medium)

**Question:** What is the write-ahead logging rule?

**Answerable from corpus:** Yes

**Expected answer:**

The log record describing a change must be made durable before the corresponding changed data page is written to persistent storage. This lets recovery use the log after a crash.

**Supporting sources:**

- `12_database_recovery_logging_wal_and_backup.pdf` (source `DBMS-12`, PDF page 2)

**Evidence excerpts:**

- `12_database_recovery_logging_wal_and_backup.pdf` p. 2: engines use more complex log formats, page identifiers, sequence numbers, checksums, and transaction state. The write-ahead logging principle requires the log record describing a change to be flushed to durable storage before the changed data page is allowed to be written. If a crash occurs after a data page was written but before its
  - Benchmark evidence-anchor SHA-256: `9bf7ece052fe7c6f99d903e5c6fb405d512dab4cc1c740ff240ddafb538dae6c`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0049 - direct_factual (easy)

**Question:** What does EXPLAIN ANALYZE do in PostgreSQL?

**Answerable from corpus:** Yes

**Expected answer:**

EXPLAIN ANALYZE actually executes the statement and reports measured execution information such as actual row counts and timing, in addition to the plan. For DML, it performs the changes.

**Supporting sources:**

- `14_query_optimization_cost_models_and_explain.pdf` (source `DBMS-14`, PDF page 5)

**Evidence excerpts:**

- `14_query_optimization_cost_models_and_explain.pdf` p. 5: Using EXPLAIN and EXPLAIN ANALYZE EXPLAIN displays the plan the optimizer intends to use and its estimates. EXPLAIN ANALYZE executes the statement and reports actual timing and row counts. For DML, EXPLAIN ANALYZE performs the changes; use a transaction and ROLLBACK in a safe test environment if you need to inspect
  - Benchmark evidence-anchor SHA-256: `2492e63e06a36d4053f7c2252251d691d3f11e4e8caa9c809106786c0d3f45cc`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0050 - direct_factual (easy)

**Question:** What are the main stages in a simplified SQL query-processing pipeline?

**Answerable from corpus:** Yes

**Expected answer:**

Parsing, semantic analysis/binding, rewriting, optimization/plan generation, and execution. The exact implementation may contain additional stages.

**Supporting sources:**

- `13_query_processing_and_execution_algorithms.pdf` (source `DBMS-13`, PDF page 1)

**Evidence excerpts:**

- `13_query_processing_and_execution_algorithms.pdf` p. 1: pipeline Query processing converts a declarative SQL statement into a physical plan and executes it. A simplified pipeline includes parsing, semantic analysis, rewrite, optimization, plan generation, and execution. Parsing checks syntax; analysis resolves names, types, and permissions; rewrite expands views
  - Benchmark evidence-anchor SHA-256: `ed3fc26c10707dacd615f638e89cb2d50621eec885129d6c9622e45a2c64461a`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0075 - conceptual_reasoning (medium)

**Question:** What problem does MVCC address, and what is the central idea?

**Answerable from corpus:** Yes

**Expected answer:**

MVCC manages multiple row versions so readers can work from a consistent snapshot while writers create newer versions. It reduces some reader–writer blocking, but it does not eliminate all locking or concurrency conflicts.

**Supporting sources:**

- `11_concurrency_control_serializability_mvcc.pdf` (source `DBMS-11`, PDF page 5)

**Evidence excerpts:**

- `11_concurrency_control_serializability_mvcc.pdf` p. 5: corpus - original explanations with cited references Page 5 5. MVCC and PostgreSQL isolation behavior Multi-Version Concurrency Control (MVCC) keeps multiple versions of rows so readers can see a consistent snapshot while writers create newer versions. Readers often avoid blocking writers, and writers can proceed
  - Benchmark evidence-anchor SHA-256: `83baf8d8a1d59f4c08b8abe7ea76d3bcc9d54ef17126f35bd705586ae1c1cf87`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0076 - conceptual_reasoning (medium)

**Question:** Why are checkpoints useful in crash recovery?

**Answerable from corpus:** Yes

**Expected answer:**

A checkpoint records a recovery position/state so recovery does not need to process the entire log history from the beginning. It helps bound crash-recovery work, but it is not a substitute for backups.

**Supporting sources:**

- `12_database_recovery_logging_wal_and_backup.pdf` (source `DBMS-12`, PDF page 3)

**Evidence excerpts:**

- `12_database_recovery_logging_wal_and_backup.pdf` p. 3: retrieval corpus - original explanations with cited references Page 3 3. Checkpoints and crash recovery A checkpoint establishes a known recovery point and records information about dirty pages and active transactions. Checkpoints reduce restart work but can create I/O pressure if too many dirty pages are flushed at once.
  - Benchmark evidence-anchor SHA-256: `7fcf3bfa54ff4a8a7b200827d8fd8214027a4e23e15e0174a7398c2e2aa51016`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0077 - conceptual_reasoning (medium)

**Question:** When is a hash join a reasonable choice compared with a nested-loop join?

**Answerable from corpus:** Yes

**Expected answer:**

A hash join is often useful for equi-joins when inputs are substantial and a hash table can be built on one input. A nested-loop join can be effective when the outer input is small and the inner side has an efficient index lookup. The optimizer considers data size and available access paths.

**Supporting sources:**

- `13_query_processing_and_execution_algorithms.pdf` (source `DBMS-13`, PDF page 3)

**Evidence excerpts:**

- `13_query_processing_and_execution_algorithms.pdf` p. 3: Educational retrieval corpus - original explanations with cited references Page 3 3. Join algorithms Nested-loop join scans an outer input and, for each outer row, searches the inner input for matches. With an index on the inner join key, this can be effective when the outer input is small. Without an efficient
  - Benchmark evidence-anchor SHA-256: `883d4f33f7cca4580db57bb3db0a5a1f2e997d05a5adc22dab05137d415bc2ea`
- `13_query_processing_and_execution_algorithms.pdf` p. 3: when the outer input is small. Without an efficient inner access path, the repeated scan can be expensive. Hash join builds a hash table from one input on the join key, then probes it with rows from the other input. It is effective for equality joins when the build side and hash structure
  - Benchmark evidence-anchor SHA-256: `9087301b1ec7fdb184cfc34cc7296bda7bda125ff4acb33d78c22c70ebe5214b`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0078 - conceptual_reasoning (medium)

**Question:** What is selection pushdown and why can it improve a query?

**Answerable from corpus:** Yes

**Expected answer:**

Selection pushdown moves a filter closer to base relations so fewer rows flow into later joins or operators. Reducing intermediate cardinality can lower CPU, memory, and I/O work while preserving results when the rewrite is valid.

**Supporting sources:**

- `14_query_optimization_cost_models_and_explain.pdf` (source `DBMS-14`, PDF page 2)

**Evidence excerpts:**

- `14_query_optimization_cost_models_and_explain.pdf` p. 2: Educational retrieval corpus - original explanations with cited references Page 2 2. Logical query rewrites A selection pushdown moves a filter closer to the base relation so fewer rows flow into later operators. A projection pushdown removes unneeded columns early, reducing tuple width and data movement. Join reordering
  - Benchmark evidence-anchor SHA-256: `cd6d98d22cd1586afc402021c102e98260f92b23d5115308719d6eedde7926f4`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0079 - conceptual_reasoning (medium)

**Question:** Why should PostgreSQL planner costs not be read as elapsed milliseconds?

**Answerable from corpus:** Yes

**Expected answer:**

Planner costs are abstract estimates in cost units for comparing alternatives, based on configured page and CPU cost parameters. They are not milliseconds; actual elapsed time must be measured separately, for example with EXPLAIN ANALYZE.

**Supporting sources:**

- `14_query_optimization_cost_models_and_explain.pdf` (source `DBMS-14`, PDF page 4)

**Evidence excerpts:**

- `14_query_optimization_cost_models_and_explain.pdf` p. 4: planner cost parameters approximate page access, CPU operator work, tuple processing, and parallel setup. These costs are not milliseconds and should not be interpreted as measured elapsed time. They help compare alternative plans under the same model. Plan selection considers startup cost and total cost. Startup
  - Benchmark evidence-anchor SHA-256: `38d1bd1caebce819de1e0c72545d18737536762ac38025bbb9b81b4292a25a00`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0080 - conceptual_reasoning (medium)

**Question:** What is the difference between replication and sharding?

**Answerable from corpus:** Yes

**Expected answer:**

Replication keeps copies of the same logical data on multiple nodes, often for availability or read capacity. Sharding horizontally distributes different subsets of rows across nodes to distribute storage and write load. A deployment can combine both.

**Supporting sources:**

- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 3)

**Evidence excerpts:**

- `15_distributed_databases_nosql_security_and_administration.pdf` p. 3: a node. A poor shard key can create hotspots, uneven storage, cross-shard joins, and difficult transactions. Replication and sharding solve different problems • Replication creates copies of the same logical data, often to improve availability or read capacity. • Sharding divides the dataset so that different nodes own different
  - Benchmark evidence-anchor SHA-256: `a0932b7b70a5a63427c1037b658109483080866af433e9359afaaf6f2972f64c`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0101 - sql_generation_explanation (medium)

**Question:** Write a query that lists the name of each course and its instructor, if an instructor is assigned.

**Answerable from corpus:** Yes

**Expected answer:**

SELECT c.course_id, c.course_title, i.full_name AS instructor_name
FROM courses AS c
LEFT JOIN instructors AS i ON i.instructor_id = c.instructor_id
ORDER BY c.course_id;

**Supporting sources:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` (source `DBMS-06`, PDF page 2)
- `university_schema.sql` (source `SQL-SCHEMA`)

**Evidence excerpts:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` p. 2: A join combines rows from related tables according to a condition. An INNER JOIN returns matching row pairs. A LEFT OUTER JOIN returns all left-side rows and matching right-side rows, using NULLs for missing right-side matches. RIGHT JOIN is symmetric; FULL OUTER JOIN preserves unmatched rows from both sides. CROSS JOIN returns
  - Benchmark evidence-anchor SHA-256: `ae8ea3cb6755e892b4d2db3b303fef865c98b516499d57742c0abf49b4dbbb57`
- `university_schema.sql` p. None: SMALLINT NOT NULL CHECK (credits BETWEEN 1 AND 6), dept_id BIGINT NOT NULL REFERENCES departments(dept_id), instructor_id BIGINT REFERENCES instructors(instructor_id) ); CREATE TABLE enrollments ( student_id BIGINT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE, course_id TEXT NOT NULL REFERENCES courses(course_id)
  - Benchmark evidence-anchor SHA-256: `6f9e7cf3bc32b07c1cfba8b3ab876f456d51b418b2aea4bc92136efc7884f83d`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0102 - sql_generation_explanation (medium)

**Question:** Write a query to show departments having more than three students.

**Answerable from corpus:** Yes

**Expected answer:**

SELECT d.dept_id, d.dept_name, COUNT(s.student_id) AS student_count
FROM departments AS d
JOIN students AS s ON s.dept_id = d.dept_id
GROUP BY d.dept_id, d.dept_name
HAVING COUNT(s.student_id) > 3;

**Supporting sources:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` (source `DBMS-06`, PDF page 3)
- `university_schema.sql` (source `SQL-SCHEMA`)

**Evidence excerpts:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` p. 3: groups sharing the grouping key. The SELECT list can include grouping columns and aggregate expressions. HAVING filters groups after aggregation; WHERE filters individual rows before grouping. SELECT course_id, COUNT(*) AS enrollment_count, AVG(grade) AS average_grade FROM enrollments GROUP BY course_id HAVING COUNT(*)
  - Benchmark evidence-anchor SHA-256: `d4176ae874800344d81bb9350dfe55bf274acc7102d62d8f833f4b1c953ee6fd`
- `university_schema.sql` p. None: departments ( dept_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, dept_name TEXT NOT NULL UNIQUE ); CREATE TABLE students ( student_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, full_name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, dept_id BIGINT REFERENCES departments(dept_id), enrollment_year
  - Benchmark evidence-anchor SHA-256: `44f3da6e47a7de3ad6e9ff874ab77836a4d49180627eea6dd515764c82526bba`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0103 - sql_generation_explanation (medium)

**Question:** Write a query that labels each enrollment grade as Pass when it is at least 40 and Needs review otherwise; preserve NULL as Missing.

**Answerable from corpus:** Yes

**Expected answer:**

SELECT student_id, course_id,
       CASE WHEN grade IS NULL THEN 'Missing'
            WHEN grade >= 40 THEN 'Pass'
            ELSE 'Needs review' END AS grade_status
FROM enrollments;

**Supporting sources:**

- `05_sql_fundamentals_ddl_dml_and_types.pdf` (source `DBMS-05`, PDF page 5)
- `university_schema.sql` (source `SQL-SCHEMA`)

**Evidence excerpts:**

- `05_sql_fundamentals_ddl_dml_and_types.pdf` p. 5: SQL expressions combine constants, column values, operators, functions, and conditional expressions. CASE implements conditional logic. COALESCE(a,b) returns the first non-NULL argument. NULLIF(a,b) returns NULL when its arguments are equal. These constructs are useful but should preserve meaning rather
  - Benchmark evidence-anchor SHA-256: `9879ff79233b3c7a0f1711bfdf62567efff228255047c02b9d17199a6cfb4f90`
- `university_schema.sql` p. None: CASCADE, course_id TEXT NOT NULL REFERENCES courses(course_id) ON DELETE RESTRICT, semester TEXT NOT NULL, grade NUMERIC(5,2) CHECK (grade BETWEEN 0 AND 100), enrolled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (student_id, course_id, semester) ); CREATE INDEX idx_students_department
  - Benchmark evidence-anchor SHA-256: `db77d48dfbd7ad0c879ecb83103dfd155d8c8b1c79e57e01a7abd15fc7b99a9e`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0104 - sql_generation_explanation (medium)

**Question:** Explain why an application should use parameterized SQL instead of concatenating an untrusted user value into a query.

**Answerable from corpus:** Yes

**Expected answer:**

Parameterized statements keep untrusted values separate from SQL syntax. This prevents user input from being interpreted as SQL code and also avoids quoting/escaping errors. Do not construct executable SQL by string-concatenating raw user input.

**Supporting sources:**

- `05_sql_fundamentals_ddl_dml_and_types.pdf` (source `DBMS-05`, PDF page 4)
- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 5)

**Evidence excerpts:**

- `05_sql_fundamentals_ddl_dml_and_types.pdf` p. 4: with the same predicate to inspect affected rows. Use transactions for related changes. In application code, use parameterized statements rather than string concatenation to prevent SQL injection and quoting errors. A database constraint can reject invalid changes even when application validation fails. TRUNCATE removes all rows efficiently
  - Benchmark evidence-anchor SHA-256: `81e43186996b14034705be1fa1321c223b264c28c8724918ca7ec16b5337a65e`
- `15_distributed_databases_nosql_security_and_administration.pdf` p. 5: and superusers can have powers beyond ordinary table privileges. SQL injection and parameterized queries SQL injection occurs when untrusted input is concatenated into SQL text and interpreted as part of the statement. Use parameterized queries or prepared statements so values remain data, not executable SQL. Identifiers
  - Benchmark evidence-anchor SHA-256: `dae79e3bf845ddce22a39b6dd4436c2dd5452baa873bc08972ebb0c87ad4fe4e`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0105 - sql_generation_explanation (hard)

**Question:** Write a query that returns each student’s name and their total number of enrollment rows, including students with zero enrollments.

**Answerable from corpus:** Yes

**Expected answer:**

SELECT s.student_id, s.full_name, COUNT(e.course_id) AS enrollment_count
FROM students AS s
LEFT JOIN enrollments AS e ON e.student_id = s.student_id
GROUP BY s.student_id, s.full_name
ORDER BY s.student_id;

**Supporting sources:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` (source `DBMS-06`, PDF page 2)
- `06_sql_queries_joins_aggregation_subqueries.pdf` (source `DBMS-06`, PDF page 3)
- `university_schema.sql` (source `SQL-SCHEMA`)

**Evidence excerpts:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` p. 2: A join combines rows from related tables according to a condition. An INNER JOIN returns matching row pairs. A LEFT OUTER JOIN returns all left-side rows and matching right-side rows, using NULLs for missing right-side matches. RIGHT JOIN is symmetric; FULL OUTER JOIN preserves unmatched rows from both sides. CROSS JOIN returns
  - Benchmark evidence-anchor SHA-256: `ae8ea3cb6755e892b4d2db3b303fef865c98b516499d57742c0abf49b4dbbb57`
- `06_sql_queries_joins_aggregation_subqueries.pdf` p. 3: summarize multiple rows. Common examples are COUNT, SUM, AVG, MIN, and MAX. COUNT(*) counts input rows; COUNT(expression) ignores NULL inputs. SUM and AVG ignore NULL values. An aggregate over no qualifying rows may produce NULL, while COUNT returns zero in the relevant scalar aggregate case. GROUP BY partitions
  - Benchmark evidence-anchor SHA-256: `2a8ca7fb3b9daf0a54e4bef015aa8c778db9658c1e12eed97811d010603fdf4d`
- `university_schema.sql` p. None: NOT NULL REFERENCES departments(dept_id), instructor_id BIGINT REFERENCES instructors(instructor_id) ); CREATE TABLE enrollments ( student_id BIGINT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE, course_id TEXT NOT NULL REFERENCES courses(course_id) ON DELETE RESTRICT, semester TEXT NOT
  - Benchmark evidence-anchor SHA-256: `f4d04f33aca0a682bf2bc89b373275c65112a0f293e4163221fbf7d07a6005c4`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0122 - cross_document_reasoning (medium)

**Question:** How do PostgreSQL roles, ordinary privileges, and row-level security provide different layers of access control?

**Answerable from corpus:** Yes

**Expected answer:**

Roles represent identities and privilege groups. GRANT/REVOKE controls permissions on database objects. Row-level security policies restrict which rows a role may see or modify. RLS complements rather than replaces ordinary privileges, authentication, credential protection, and application authorization.

**Supporting sources:**

- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 5)
- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 6)
- `05_sql_fundamentals_ddl_dml_and_types.pdf` (source `DBMS-05`, PDF page 1)

**Evidence excerpts:**

- `15_distributed_databases_nosql_security_and_administration.pdf` p. 5: explanations with cited references Page 5 5. PostgreSQL security: identities, privileges, and safe SQL PostgreSQL uses roles to represent database identities and privilege groups. A role can own objects, receive privileges, and—when configured appropriately—log in. Use separate roles for application access,
  - Benchmark evidence-anchor SHA-256: `d952fb85731e3e1b20e31bb5d07ce8bb98f78875fbe7d5256b8ffebbe19e6037`
- `15_distributed_databases_nosql_security_and_administration.pdf` p. 6: - original explanations with cited references Page 6 6. Row security, administration, and review PostgreSQL row-level security (RLS) can restrict which rows a role may see or modify. Table owners, superusers, and roles with BYPASSRLS have special behavior; policy design must account for it. RLS complements ordinary
  - Benchmark evidence-anchor SHA-256: `53ea6e0797579ef7e503e47d3d16b5fc5b52a8689c404a957635b649794d7d13`
- `05_sql_fundamentals_ddl_dml_and_types.pdf` p. 1: across books and products. DDL defines objects such as tables and indexes. DML retrieves and changes rows. DCL manages privileges. TCL controls transaction boundaries. SQL also includes constraints, views, routines, and other features. Core command groups • DDL: CREATE, ALTER, DROP. • DML: SELECT, INSERT,
  - Benchmark evidence-anchor SHA-256: `ab49d4062885c0c19e275cb27f727d5b395b8a34d5be710fc8d98e4b67dd4593`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0123 - cross_document_reasoning (medium)

**Question:** Why does choosing NUMERIC for an exact grade/financial value complement CHECK constraints?

**Answerable from corpus:** Yes

**Expected answer:**

NUMERIC stores exact decimal precision and scale; a CHECK constraint enforces a permitted range or other invariant. Choosing a suitable data type controls representation, while the constraint controls which values are acceptable. If the field is mandatory, NOT NULL is also needed.

**Supporting sources:**

- `05_sql_fundamentals_ddl_dml_and_types.pdf` (source `DBMS-05`, PDF page 2)
- `02_relational_model_keys_and_integrity.pdf` (source `DBMS-02`, PDF page 3)

**Evidence excerpts:**

- `05_sql_fundamentals_ddl_dml_and_types.pdf` p. 2: numbers. Common PostgreSQL types • SMALLINT, INTEGER, BIGINT: signed whole numbers with different ranges. • NUMERIC(p,s): exact decimal precision and scale; useful for financial or grading quantities. • REAL and DOUBLE PRECISION: approximate floating-point values. • TEXT and VARCHAR(n): variable-length character data;
  - Benchmark evidence-anchor SHA-256: `73d7061a26ff5bf883f27b237fdee3837094d4f4cf5a4f57d0eef8f1a8dc8569`
- `02_relational_model_keys_and_integrity.pdf` p. 3: unique, non-null key. • FOREIGN KEY: ensures references point to rows in another table or the same table. • CHECK: requires a condition to be true for each inserted or updated row. A CHECK evaluating to NULL is not rejected, so use NOT NULL separately when required. • DEFAULT: supplies a value when an INSERT omits
  - Benchmark evidence-anchor SHA-256: `0e7e1050aba2be9258d12dac11c1dda5cdba04fc6cbc6eace792611be7b90e2b`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0124 - cross_document_reasoning (medium)

**Question:** How do the logical query-processing order and physical execution plan differ?

**Answerable from corpus:** Yes

**Expected answer:**

The logical model explains query meaning—for example FROM/JOIN, WHERE, GROUP BY, HAVING, SELECT, DISTINCT, ORDER BY, LIMIT. A physical plan is the optimizer-selected algorithm/tree (scans, joins, sorts, aggregates) used to produce the same logical result.

**Supporting sources:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` (source `DBMS-06`, PDF page 1)
- `13_query_processing_and_execution_algorithms.pdf` (source `DBMS-13`, PDF page 1)

**Evidence excerpts:**

- `06_sql_queries_joins_aggregation_subqueries.pdf` p. 1: The DBMS may reorder operations, choose indexes, or use parallel workers while preserving query semantics. A simplified logical processing model is: FROM and JOIN construct a source relation; WHERE filters rows; GROUP BY forms groups; HAVING filters groups; SELECT computes output expressions; DISTINCT removes duplicates;
  - Benchmark evidence-anchor SHA-256: `0e332c4e517d5447592033d1bb3452a80a15d1eaecca13aedeca7a83d83a6a15`
- `13_query_processing_and_execution_algorithms.pdf` p. 1: expands views and rules; the optimizer compares candidate plans; the executor runs physical operators. A query plan is commonly represented as a tree or graph of operators. Leaves access base tables or indexes. Intermediate operators filter, project, join, sort, and aggregate rows. A plan is a physical implementation
  - Benchmark evidence-anchor SHA-256: `f11c01b71673217c65090e8da5c530d02ae0ad95da10446e1f011d93a58a1108`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0125 - cross_document_reasoning (medium)

**Question:** How do a backup and WAL serve different roles in database recovery?

**Answerable from corpus:** Yes

**Expected answer:**

A backup supplies a restorable copy of database contents, especially after media loss. WAL records changes and supports crash recovery or point-in-time recovery when combined with the required base backup and archived log sequence. A checkpoint alone is not a backup.

**Supporting sources:**

- `12_database_recovery_logging_wal_and_backup.pdf` (source `DBMS-12`, PDF page 4)
- `12_database_recovery_logging_wal_and_backup.pdf` (source `DBMS-12`, PDF page 5)
- `10_transactions_and_acid_properties.pdf` (source `DBMS-10`, PDF page 2)

**Evidence excerpts:**

- `12_database_recovery_logging_wal_and_backup.pdf` p. 4: retrieval corpus - original explanations with cited references Page 4 4. PostgreSQL WAL and durability PostgreSQL uses Write-Ahead Logging. WAL records changes to database pages. The server can flush WAL to durable storage and later write changed data pages. During crash recovery, PostgreSQL replays WAL as
  - Benchmark evidence-anchor SHA-256: `564b04e160828016537cdb82bdf769376996e801d00b92b612819b85a092dca8`
- `12_database_recovery_logging_wal_and_backup.pdf` p. 5: corpus - original explanations with cited references Page 5 5. Backups, restore, and point-in-time recovery A backup strategy defines how to recover from data loss, corruption, accidental deletion, and site failure. Logical backups export SQL or logical objects; physical backups copy the database cluster files in a consistent
  - Benchmark evidence-anchor SHA-256: `cd8e217503f4d753ad93cb089a753fdca812cfdd4e53147c1ffb515ff9841a34`
- `10_transactions_and_acid_properties.pdf` p. 2: concurrency, and performance. The exact phenomena permitted depend on the DBMS and isolation implementation. Durability: once a commit is acknowledged under the configured durability guarantees, the committed result should survive specified failures. Logging, flush policy, storage guarantees, replication, and configuration
  - Benchmark evidence-anchor SHA-256: `d65259bf995b40280fd91a83ffa3de1c79e0e3c1fa14310ab807f48b39faf88f`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0138 - scenario_based (hard)

**Question:** A multi-tenant app uses row-level security based on a session variable. What critical security assumption must be checked?

**Answerable from corpus:** Yes

**Expected answer:**

The application must set the session identity through a trusted mechanism. If an untrusted caller can freely set that session variable, it is not authentication and the caller may impersonate another tenant. Pooled connections must also reset session state safely.

**Supporting sources:**

- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 6)

**Evidence excerpts:**

- `15_distributed_databases_nosql_security_and_administration.pdf` p. 6: ON enrollments FOR SELECT TO student_app USING (student_id = current_setting('app.student_id')::bigint); This example requires a trusted mechanism to set the session identity. If a caller can freely set the session variable, the policy is not an authentication mechanism. In pooled connections, session state must be
  - Benchmark evidence-anchor SHA-256: `55a3d962848f39a035c70cfe1936e9bc93d1b25233426f1097378ccda69b5b1d`
- `15_distributed_databases_nosql_security_and_administration.pdf` p. 6: student_id')::bigint); This example requires a trusted mechanism to set the session identity. If a caller can freely set the session variable, the policy is not an authentication mechanism. In pooled connections, session state must be reset safely. Test policies using the actual application role, not only the
  - Benchmark evidence-anchor SHA-256: `2cce1dfaf356122403028c42463ff5460338b00f89e59f53fd42f2e258cd63e2`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0139 - scenario_based (medium)

**Question:** A social application frequently traverses friends-of-friends relationships. Which database model may fit this access pattern better than a key-value store?

**Answerable from corpus:** Yes

**Expected answer:**

A graph database may fit highly connected data and relationship traversal, because it represents vertices and edges and supports graph-oriented queries. The choice still depends on graph shape, query patterns, and the engine.

**Supporting sources:**

- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 2)

**Evidence excerpts:**

- `15_distributed_databases_nosql_security_and_administration.pdf` p. 2: across nodes, but schema and key design must avoid hotspots and support the queries the application needs. Graph databases represent vertices (entities) and edges (relationships), often with properties. They can be suitable for highly connected data such as social networks, recommendations, fraud patterns, and
  - Benchmark evidence-anchor SHA-256: `9fc6cfd6d53d8b838a411cf3e75657840e13b15b741583170ebd7d37d8f0a708`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0140 - scenario_based (medium)

**Question:** A read replica is asynchronous. Immediately after a user updates a record on the primary, the next read from the replica shows the old value. Is this necessarily a failed write?

**Answerable from corpus:** Yes

**Expected answer:**

Not necessarily. Asynchronous replication can lag, so a replica read immediately after a primary write may return stale data. The system should route freshness-sensitive reads appropriately or use a consistency policy that waits for the required replication state.

**Supporting sources:**

- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 3)

**Evidence excerpts:**

- `15_distributed_databases_nosql_security_and_administration.pdf` p. 3: propagates changes to replicas. Replicas can serve reads, support failover, or provide geographic copies. With asynchronous replication, a replica may lag behind the primary; a client that reads immediately from a lagging replica may not see its latest write. Synchronous replication waits for specified replicas before confirming
  - Benchmark evidence-anchor SHA-256: `59a84e78d0faae9a2c2e6140074d34c89978d2c4bad7e7d8ef5d75a8809b4343`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0149 - unanswerable (hard)

**Question:** Which ClickHouse compression codec is optimal for every time-series workload, and what exact compression ratio will it achieve?

**Answerable from corpus:** No

**Expected answer:**

The corpus does not cover ClickHouse codecs or workload-specific compression measurements. No universal optimal codec or exact compression ratio is supported; the assistant should request benchmark conditions and authoritative documentation.

**Evidence:** No supporting passage is present in the supplied corpus. The expected behavior is to state that the corpus does not provide enough information and avoid inventing the requested details.

**Scoring note:** Expected behavior: abstain or state that the indexed corpus does not contain enough evidence; do not guess.

---

## DBMS-EVAL-0150 - unanswerable (hard)

**Question:** Give the exact failover procedure for PostgreSQL 18 logical replication slots using the newest failover-slot synchronization behavior.

**Answerable from corpus:** No

**Expected answer:**

The corpus only introduces replication generically and does not document the specified PostgreSQL logical-replication-slot failover procedure. Because the question asks for precise version-specific operational steps, the assistant should retrieve the matching PostgreSQL 18 documentation or abstain.

**Evidence:** No supporting passage is present in the supplied corpus. The expected behavior is to state that the corpus does not provide enough information and avoid inventing the requested details.

**Scoring note:** Expected behavior: abstain or state that the indexed corpus does not contain enough evidence; do not guess.

---

## DBMS-EVAL-0159 - negation_scope_hard_negative (hard)

**Question:** Does PostgreSQL write a changed data page to persistent storage before making the corresponding WAL record durable?

**Answerable from corpus:** Yes

**Expected answer:**

No. The write-ahead logging rule requires the relevant log record to be durable before the changed data page is written. This ordering is necessary for recovery.

**Supporting sources:**

- `12_database_recovery_logging_wal_and_backup.pdf` (source `DBMS-12`, PDF page 2)

**Evidence excerpts:**

- `12_database_recovery_logging_wal_and_backup.pdf` p. 2: engines use more complex log formats, page identifiers, sequence numbers, checksums, and transaction state. The write-ahead logging principle requires the log record describing a change to be flushed to durable storage before the changed data page is allowed to be written. If a crash occurs after a data page was written but before its
  - Benchmark evidence-anchor SHA-256: `9bf7ece052fe7c6f99d903e5c6fb405d512dab4cc1c740ff240ddafb538dae6c`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

## DBMS-EVAL-0160 - negation_scope_hard_negative (hard)

**Question:** Does asynchronous replication guarantee that a read from a replica immediately after a primary write returns the newest value?

**Answerable from corpus:** Yes

**Expected answer:**

No. An asynchronous replica can lag behind the primary, so an immediate replica read may return an older value. The read path must account for the consistency/freshness requirement.

**Supporting sources:**

- `15_distributed_databases_nosql_security_and_administration.pdf` (source `DBMS-15`, PDF page 3)

**Evidence excerpts:**

- `15_distributed_databases_nosql_security_and_administration.pdf` p. 3: propagates changes to replicas. Replicas can serve reads, support failover, or provide geographic copies. With asynchronous replication, a replica may lag behind the primary; a client that reads immediately from a lagging replica may not see its latest write. Synchronous replication waits for specified replicas before confirming
  - Benchmark evidence-anchor SHA-256: `59a84e78d0faae9a2c2e6140074d34c89978d2c4bad7e7d8ef5d75a8809b4343`

**Scoring note:** Score semantic correctness and evidence support; accept equivalent wording, not unsupported extra claims.

---

