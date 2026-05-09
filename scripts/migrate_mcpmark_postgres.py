#!/usr/bin/env python3
"""Migrate remaining 14 mcpmark postgres tasks to Bella cases (008-021)."""

import json
import sqlite3
from pathlib import Path

CASES_DIR = Path("cases")
ENV_DIR = Path("environments")


def write_case(case):
    path = CASES_DIR / f"{case['case_id']}.json"
    with open(path, "w") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)
    print(f"  {case['case_id']}: {path.name}")


def q(db, sql):
    conn = sqlite3.connect(str(ENV_DIR / f"mcpmark_postgres_{db}" / "world" / "world.db"))
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [list(r) for r in rows]


def base(case_id, env_db, tags, description):
    return {
        "case_id": case_id,
        "env_name": f"mcpmark_postgres_{env_db}",
        "category": "mcpmark_postgres",
        "source": "mcpmark",
        "tags": tags,
        "interaction_mode": "fixed",
        "user_demands": [description],
        "world_setup": [],
    }


def main():
    # ===== 008: employee_projects_basic =====
    c = base("mcpmark_postgres_008", "employees",
             ["schema design", "data insertion", "difficulty:L1", "database:employees"],
             "Create and manage a basic employee projects table to track company projects.\n\n"
             "## Your Tasks:\n\n"
             "1. **Create the employee_projects table** with:\n"
             "   * `project_id` (integer, primary key, autoincrement)\n"
             "   * `project_name` (varchar(100), not null)\n"
             "   * `start_date` (text/date, not null)\n"
             "   * `end_date` (text/date)\n"
             "   * `budget` (real/decimal)\n"
             "   * `status` (varchar(20), default 'active')\n\n"
             "2. **Insert exactly this initial data**:\n"
             "   * Project 1: name='Database Modernization', start_date='2024-01-15', end_date='2024-06-30', budget=250000.00, status='active'\n"
             "   * Project 2: name='Employee Portal Upgrade', start_date='2024-02-01', end_date='2024-05-15', budget=180000.00, status='active'\n"
             "   * Project 3: name='HR Analytics Dashboard', start_date='2023-11-01', end_date='2024-01-31', budget=120000.00, status='active'")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM employee_projects", "expected": [[3]], "order_matters": False},
        {"sql": "SELECT project_name, budget, status FROM employee_projects ORDER BY project_name",
         "expected": [
             ["Database Modernization", 250000.0, "active"],
             ["Employee Portal Upgrade", 180000.0, "active"],
             ["HR Analytics Dashboard", 120000.0, "active"],
         ], "order_matters": True},
    ]
    write_case(c)

    # ===== 009: hiring_year_summary =====
    expected_rows = q("employees", """
        WITH current_emp AS (
            SELECT DISTINCT employee_id FROM salary WHERE to_date = '9999-01-01'
        )
        SELECT
            CAST(strftime('%Y', e.hire_date) AS INTEGER) AS hire_year,
            COUNT(*) AS employees_hired,
            SUM(CASE WHEN ce.employee_id IS NOT NULL THEN 1 ELSE 0 END) AS still_employed
        FROM employee e
        LEFT JOIN current_emp ce ON ce.employee_id = e.id
        WHERE e.hire_date IS NOT NULL
        GROUP BY hire_year
        ORDER BY hire_year
    """)
    n_years = len(expected_rows)
    spot = expected_rows[0]  # first year
    c = base("mcpmark_postgres_009", "employees",
             ["reporting and analytics", "data aggregation", "difficulty:L1", "database:employees"],
             "Create a hiring year summary table to help HR track employee retention trends.\n\n"
             "## Your Task:\n\n"
             "**Create a table called `hiring_year_summary`** with these exact columns:\n\n"
             "* `hire_year` (integer) — year employees were hired\n"
             "* `employees_hired` (integer) — number of employees hired that year\n"
             "* `still_employed` (integer) — how many from that year are still employed (have active salary where to_date = '9999-01-01')\n"
             "* `retention_rate` (real) — percentage still employed (still_employed / employees_hired * 100)\n\n"
             "## Requirements:\n\n"
             "1. Extract the hire year from the `hire_date` column in the `employee` table (use strftime('%Y', hire_date))\n"
             "2. Count total employees hired in each year\n"
             "3. Determine which employees are still employed by checking for active salary records (to_date = '9999-01-01' in the `salary` table)\n"
             "4. Order results by hire_year in ascending order")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM hiring_year_summary", "expected": [[n_years]], "order_matters": False},
        {"sql": "SELECT hire_year, employees_hired, still_employed FROM hiring_year_summary ORDER BY hire_year LIMIT 1",
         "expected": [spot[:3]], "order_matters": True},
    ]
    write_case(c)

    # ===== 010: employee_demographics_report =====
    gender_expected = q("employees", """
        WITH current_emp AS (SELECT DISTINCT employee_id FROM salary WHERE to_date='9999-01-01')
        SELECT gender, COUNT(*), SUM(CASE WHEN ce.employee_id IS NOT NULL THEN 1 ELSE 0 END)
        FROM employee e LEFT JOIN current_emp ce ON ce.employee_id=e.id
        WHERE gender IN ('M','F') GROUP BY gender ORDER BY gender
    """)
    c = base("mcpmark_postgres_010", "employees",
             ["reporting and analytics", "demographics", "difficulty:L3", "database:employees"],
             "Generate a comprehensive employee demographics report. Create these 4 tables:\n\n"
             "### 1. `gender_statistics`\n"
             "* `gender` (varchar), `total_employees` (integer), `current_employees` (integer), `percentage_of_workforce` (real)\n"
             "* Count all employees by gender from `employee` table\n"
             "* Current = has active salary (to_date='9999-01-01')\n\n"
             "### 2. `age_group_analysis`\n"
             "* `age_group` (varchar: '20-29','30-39','40-49','50-59','60+'), `employee_count` (integer), `avg_salary` (real), `avg_tenure_days` (real)\n"
             "* Only current employees. Use date('now') for current date.\n"
             "* Age = years between birth_date and now. Tenure = days between hire_date and now.\n\n"
             "### 3. `birth_month_distribution`\n"
             "* `birth_month` (integer 1-12), `month_name` (varchar), `employee_count` (integer), `current_employee_count` (integer)\n"
             "* Include all 12 months.\n\n"
             "### 4. `hiring_year_summary`\n"
             "* `hire_year` (integer), `employees_hired` (integer), `still_employed` (integer), `retention_rate` (real)\n"
             "* Extract year with strftime('%Y', hire_date). Current = has active salary.")
    c["verify"] = [
        {"sql": "SELECT gender, total_employees, current_employees FROM gender_statistics ORDER BY gender",
         "expected": gender_expected, "order_matters": True},
        {"sql": "SELECT COUNT(*) FROM age_group_analysis", "expected": [[5]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM birth_month_distribution", "expected": [[12]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM hiring_year_summary WHERE hire_year IS NOT NULL",
         "expected": [[n_years]], "order_matters": False},
    ]
    write_case(c)

    # ===== 011: employee_performance_analysis =====
    dept_salary = q("employees", """
        WITH current_salary AS (
            SELECT employee_id, amount FROM (
                SELECT s.*, ROW_NUMBER() OVER (PARTITION BY s.employee_id ORDER BY s.from_date DESC, s.amount DESC) AS rn
                FROM salary s WHERE s.to_date='9999-01-01'
            ) x WHERE rn=1
        ),
        current_dept AS (
            SELECT DISTINCT de.employee_id, de.department_id FROM department_employee de WHERE de.to_date='9999-01-01'
        )
        SELECT d.dept_name, ROUND(AVG(cs.amount),2), COUNT(cd.employee_id),
               MAX(cs.amount)-MIN(cs.amount)
        FROM department d
        JOIN current_dept cd ON cd.department_id=d.id
        JOIN current_salary cs ON cs.employee_id=cd.employee_id
        GROUP BY d.id, d.dept_name ORDER BY d.dept_name
    """)
    c = base("mcpmark_postgres_011", "employees",
             ["analytics", "performance evaluation", "difficulty:L3", "database:employees"],
             "Create a comprehensive employee performance evaluation system.\n\n"
             "### 1. `employee_performance_analysis` table\n"
             "* `employee_id` (integer), `performance_category` (varchar), `salary_growth_rate` (real), `days_of_service` (integer), `promotion_count` (integer)\n"
             "* Only current employees (active salary where to_date='9999-01-01')\n"
             "* salary_growth_rate = ((current_salary - first_salary) / first_salary) * 100\n"
             "* days_of_service = julianday('now') - julianday(hire_date) (cast to integer)\n"
             "* promotion_count = number of distinct titles held\n"
             "* Performance categories:\n"
             "  - 'high_achiever': growth > 40% AND promotion_count > 1\n"
             "  - 'needs_attention': growth < 15% AND days_of_service > 3650\n"
             "  - 'steady_performer': all others\n\n"
             "### 2. `department_salary_analysis` table\n"
             "* `department_name` (varchar), `avg_current_salary` (real), `employee_count` (integer), `salary_range_spread` (integer)\n"
             "* Only current employees. salary_range_spread = max salary - min salary in department.\n"
             "* Order by department_name.")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM employee_performance_analysis WHERE performance_category IN ('high_achiever','steady_performer','needs_attention')",
         "expected": [[q("employees", "SELECT COUNT(DISTINCT employee_id) FROM salary WHERE to_date='9999-01-01'")[0][0]]],
         "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM department_salary_analysis", "expected": [[9]], "order_matters": False},
        {"sql": "SELECT department_name, employee_count FROM department_salary_analysis ORDER BY department_name LIMIT 3",
         "expected": [[r[0], r[2]] for r in dept_salary[:3]], "order_matters": True},
    ]
    write_case(c)

    # ===== 012: employee_project_tracking =====
    # Count current employees per department for assignment verification
    dept_counts = q("employees", """
        SELECT d.dept_name, COUNT(DISTINCT de.employee_id)
        FROM department d JOIN department_employee de ON de.department_id=d.id
        WHERE de.to_date='9999-01-01'
        GROUP BY d.id, d.dept_name ORDER BY d.dept_name
    """)
    total_assignments = sum(r[1] for r in dept_counts)
    c = base("mcpmark_postgres_012", "employees",
             ["schema design", "data manipulation", "difficulty:L3", "database:employees"],
             "Create a comprehensive employee project tracking system.\n\n"
             "### 1. Create three tables:\n\n"
             "**`employee_projects`**: project_id (INTEGER PRIMARY KEY), project_name (VARCHAR(100) NOT NULL), start_date (TEXT NOT NULL), end_date (TEXT), budget (REAL), status (VARCHAR(20) DEFAULT 'active')\n\n"
             "**`project_assignments`**: assignment_id (INTEGER PRIMARY KEY), employee_id (INTEGER NOT NULL), project_id (INTEGER NOT NULL), role (VARCHAR(50) NOT NULL), allocation_percentage (INTEGER CHECK(allocation_percentage BETWEEN 1 AND 100)), assigned_date (TEXT NOT NULL)\n\n"
             "**`project_milestones`**: milestone_id (INTEGER PRIMARY KEY), project_id (INTEGER NOT NULL), milestone_name (VARCHAR(100) NOT NULL), due_date (TEXT NOT NULL), completed (INTEGER DEFAULT 0)\n\n"
             "### 2. Create indexes:\n"
             "* `idx_projects_status` on employee_projects(status)\n"
             "* `idx_assignments_emp_proj` on project_assignments(employee_id, project_id)\n"
             "* `idx_milestones_due_date` on project_milestones(due_date)\n\n"
             "### 3. Insert data:\n\n"
             "**employee_projects** (3 projects):\n"
             "* 'Database Modernization', '2024-01-15', '2024-06-30', 250000.00, 'active'\n"
             "* 'Employee Portal Upgrade', '2024-02-01', '2024-05-15', 180000.00, 'active'\n"
             "* 'HR Analytics Dashboard', '2023-11-01', '2024-01-31', 120000.00, 'active'\n\n"
             "**project_assignments**: Assign ALL current employees (to_date='9999-01-01' in department_employee) by department:\n"
             "* Development → Project 1 (Database Modernization), role='Developer', allocation=80%\n"
             "* Human Resources → Project 2 (Employee Portal Upgrade), role='Business Analyst', allocation=60%\n"
             "* Marketing → Project 3 (HR Analytics Dashboard), role='Marketing Specialist', allocation=40%\n"
             "* Finance → Project 1, role='Financial Analyst', allocation=30%\n"
             "* Sales → Project 2, role='Sales Representative', allocation=50%\n"
             "* Research → Project 3, role='Research Analyst', allocation=70%\n"
             "* Production → Project 1, role='Production Coordinator', allocation=45%\n"
             "* Quality Management → Project 2, role='QA Specialist', allocation=85%\n"
             "* Customer Service → Project 3, role='Customer Success', allocation=35%\n"
             "* All assigned_date='2024-01-01'\n\n"
             "**project_milestones** (6 milestones):\n"
             "* Project 1: 'Design Phase Complete' due '2024-03-01', 'Implementation Complete' due '2024-05-15'\n"
             "* Project 2: 'UI/UX Approval' due '2024-03-15', 'Beta Testing' due '2024-04-30'\n"
             "* Project 3: 'Data Collection' due '2023-12-15', 'Dashboard Launch' due '2024-01-25'\n\n"
             "### 4. Perform updates:\n"
             "* Update Project 3 status to 'completed'\n"
             "* Increase budget by 15% for all 'active' projects\n"
             "* Mark 'Data Collection' milestone as completed (set completed=1)\n\n"
             "### 5. Add priority column:\n"
             "* Add `priority` column (VARCHAR(10)) to employee_projects\n"
             "* Set priority='high' for 'Database Modernization', 'medium' for others")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM employee_projects", "expected": [[3]], "order_matters": False},
        {"sql": "SELECT project_name, ROUND(budget,2), status, priority FROM employee_projects ORDER BY project_name",
         "expected": [
             ["Database Modernization", 287500.0, "active", "high"],
             ["Employee Portal Upgrade", 207000.0, "active", "medium"],
             ["HR Analytics Dashboard", 120000.0, "completed", "medium"],
         ], "order_matters": True},
        {"sql": "SELECT COUNT(*) FROM project_assignments", "expected": [[total_assignments]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM project_milestones", "expected": [[6]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM project_milestones WHERE completed=1", "expected": [[1]], "order_matters": False},
    ]
    write_case(c)

    # ===== 013: employee_retention_analysis =====
    c = base("mcpmark_postgres_013", "employees",
             ["analytics", "retention", "difficulty:L3", "database:employees"],
             "Analyze employee retention patterns.\n\n"
             "### 1. `employee_retention_analysis` table\n"
             "* `department_name` (varchar), `total_employees_ever` (integer), `current_employees` (integer), `former_employees` (integer), `retention_rate` (real)\n"
             "* total_employees_ever = all employees ever in this department (from department_employee)\n"
             "* current_employees = those with to_date='9999-01-01' in department_employee\n"
             "* former_employees = total - current\n"
             "* retention_rate = current/total * 100\n\n"
             "### 2. `high_risk_employees` table\n"
             "* `employee_id` (integer), `full_name` (varchar), `current_department` (varchar), `tenure_days` (integer), `current_salary` (integer), `risk_category` (varchar)\n"
             "* Only current employees (active salary to_date='9999-01-01')\n"
             "* tenure_days = julianday(date('now')) - julianday(hire_date)\n"
             "* Risk categories (check against the department's retention_rate from table 1):\n"
             "  - 'high_risk': department retention_rate < 80% AND tenure < 1095 days\n"
             "  - 'medium_risk': department retention_rate < 85% AND tenure < 1825 days\n"
             "  - 'low_risk': all others\n\n"
             "### 3. `turnover_trend_analysis` table\n"
             "* `departure_year` (integer), `departures_count` (integer), `avg_tenure_days` (real), `avg_final_salary` (real)\n"
             "* Analyze employees who left between 1985-2002\n"
             "* departure_year = year extracted from to_date of salary records (use strftime('%Y', to_date))\n"
             "* For final salary: if an employee has multiple salary records with the same departure date, use the one with the latest from_date; if still tied, use the highest amount\n"
             "* avg_tenure_days = average of (julianday(to_date) - julianday(hire_date)) for departed employees")
    retention = q("employees", """
        SELECT d.dept_name,
               COUNT(DISTINCT de.employee_id),
               COUNT(DISTINCT CASE WHEN de.to_date='9999-01-01' THEN de.employee_id END)
        FROM department d JOIN department_employee de ON de.department_id=d.id
        GROUP BY d.id, d.dept_name ORDER BY d.dept_name
    """)
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM employee_retention_analysis", "expected": [[9]], "order_matters": False},
        {"sql": "SELECT department_name, total_employees_ever, current_employees FROM employee_retention_analysis ORDER BY department_name LIMIT 3",
         "expected": [r[:3] for r in retention[:3]], "order_matters": True},
        {"sql": "SELECT COUNT(*) FROM high_risk_employees WHERE risk_category IN ('high_risk','medium_risk','low_risk')",
         "expected": [[q("employees","SELECT COUNT(DISTINCT employee_id) FROM salary WHERE to_date='9999-01-01'")[0][0]]],
         "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM turnover_trend_analysis WHERE departure_year BETWEEN 1985 AND 2002",
         "expected": [[q("employees","SELECT COUNT(DISTINCT CAST(strftime('%Y',to_date) AS INTEGER)) FROM salary WHERE to_date!='9999-01-01' AND CAST(strftime('%Y',to_date) AS INTEGER) BETWEEN 1985 AND 2002")[0][0]]],
         "order_matters": False},
    ]
    write_case(c)

    # ===== 014: management_structure_analysis =====
    mgr_count = q("employees", "SELECT COUNT(DISTINCT employee_id) FROM department_manager")[0][0]
    c = base("mcpmark_postgres_014", "employees",
             ["analytics", "management structure", "difficulty:L3", "database:employees"],
             "Conduct a comprehensive management structure analysis.\n\n"
             "### 1. `manager_profile` table\n"
             "* `manager_id` (integer), `manager_name` (varchar — first_name || ' ' || last_name), `current_department` (varchar — NULL if not current), `management_periods` (integer — total management assignments including multiple periods), `current_manager` (integer — 1 if currently a manager, 0 otherwise)\n"
             "* Include all employees who have ever been a department manager\n\n"
             "### 2. `department_leadership` table\n"
             "* `department_name` (varchar), `current_manager_name` (varchar), `manager_start_date` (text), `total_historical_managers` (integer)\n"
             "* Current manager = to_date='9999-01-01' in department_manager\n\n"
             "### 3. `management_transitions` table\n"
             "* `department_name` (varchar), `transition_year` (integer), `outgoing_manager` (varchar), `incoming_manager` (varchar — 'No Successor' if none), `transition_gap_days` (integer — 0 if immediate or no successor)\n"
             "* Track when managers changed in each department\n\n"
             "### 4. `span_of_control` table\n"
             "* `manager_id` (integer), `manager_name` (varchar), `department_name` (varchar), `total_employees` (integer — all employees ever), `current_employees` (integer — active employees), `management_load` (varchar)\n"
             "* Only current managers (to_date='9999-01-01')\n"
             "* Load: 'light' < 5000, 'moderate' 5000-15000, 'heavy' > 15000")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM manager_profile", "expected": [[mgr_count]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM department_leadership", "expected": [[9]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM span_of_control", "expected": [[9]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM management_transitions WHERE transition_year IS NOT NULL",
         "expected": [[q("employees","""
             SELECT COUNT(*) FROM (
                 SELECT department_id, from_date,
                        LAG(employee_id) OVER (PARTITION BY department_id ORDER BY from_date) as prev
                 FROM department_manager
             ) WHERE prev IS NOT NULL
         """)[0][0]]], "order_matters": False},
    ]
    write_case(c)

    # ===== 015: customer_data_migration (200 customers) =====
    with open("/tmp/chinook_customers_200.json") as f:
        customers = json.load(f)
    # Build markdown table for description
    cust_table = "| FirstName | LastName | Company | Address | City | State | Country | PostalCode | Phone | Email |\n"
    cust_table += "|-----------|----------|---------|---------|------|-------|---------|------------|-------|--------|\n"
    for cu in customers:
        cust_table += f"| {cu['FirstName']} | {cu['LastName']} | {cu['Company']} | {cu['Address']} | {cu['City']} | {cu['State']} | {cu['Country']} | {cu['PostalCode']} | {cu['Phone']} | {cu['Email']} |\n"

    c = base("mcpmark_postgres_015", "chinook",
             ["data migration", "bulk operations", "difficulty:L3", "database:chinook"],
             'Migrate 200 customer records from an acquired company into the database.\n\n'
             '## Mission:\n\nChinook Music Store has acquired "MelodyMart." Migrate their customer database.\n\n'
             '## Requirements:\n\n'
             '1. Insert all customer records below into the `Customer` table\n'
             '2. Assign `CustomerId` starting from 60 (next available ID after max 59)\n'
             '3. Set `SupportRepId` = 3 for all migrated customers\n'
             '4. Set `Fax` = NULL for all migrated customers\n\n'
             '## Customer Data:\n\n' + cust_table)
    c["verify"] = [
        {"sql": 'SELECT COUNT(*) FROM "Customer" WHERE "CustomerId" > 59', "expected": [[200]], "order_matters": False},
        {"sql": 'SELECT "FirstName","LastName","SupportRepId","Fax" FROM "Customer" WHERE "CustomerId"=60',
         "expected": [["Danielle", "Johnson", 3, None]], "order_matters": False},
        {"sql": 'SELECT COUNT(*) FROM "Customer" WHERE "CustomerId">59 AND "SupportRepId"=3 AND "Fax" IS NULL',
         "expected": [[200]], "order_matters": False},
    ]
    write_case(c)

    # ===== 016: employee_hierarchy_management =====
    c = base("mcpmark_postgres_016", "chinook",
             ["CRUD operations", "data manipulation", "difficulty:L3", "database:chinook"],
             "Manage employee hierarchy and customer assignments through systematic operations.\n\n"
             "## Tasks (complete in order):\n\n"
             "### 1. INSERT: Add New Employees\n"
             "* EmployeeId=9, FirstName='Sarah', LastName='Johnson', Title='Sales Support Agent', ReportsTo=2, BirthDate='1985-03-15', HireDate='2009-01-10', Address='123 Oak Street', City='Calgary', State='AB', Country='Canada', PostalCode='T2P 5G3', Phone='+1 (403) 555-0123', Fax='+1 (403) 555-0124', Email='sarah.johnson@chinookcorp.com'\n"
             "* EmployeeId=10, FirstName='Mike', LastName='Chen', Title='Sales Support Agent', ReportsTo=2, BirthDate='1982-08-22', HireDate='2009-01-10', Address='456 Pine Ave', City='Calgary', State='AB', Country='Canada', PostalCode='T2P 5G4', Phone='+1 (403) 555-0125', Fax='+1 (403) 555-0126', Email='mike.chen@chinookcorp.com'\n\n"
             "### 2. UPDATE: Modify Employees\n"
             "* Andrew Adams (EmployeeId=1): Title → 'CEO'\n"
             "* Nancy Edwards (EmployeeId=2): Phone → '+1 (403) 555-9999'\n"
             "* All employees with Title='IT Staff' → Title='IT Specialist'\n\n"
             "### 3. UPDATE: Reassign Customers\n"
             "* CustomerId 1,2,3 → SupportRepId=9 (Sarah)\n"
             "* CustomerId 4,5,6 → SupportRepId=10 (Mike)\n\n"
             "### 4. UPDATE: Reporting Structure\n"
             "* Sarah (9) and Mike (10) → ReportsTo=1 (Andrew Adams)\n\n"
             "### 5. CREATE: employee_performance table\n"
             "* employee_id (integer), customers_assigned (integer), performance_score (real)\n"
             "* Insert for employee 9: customers_assigned=actual count, performance_score=4.5\n"
             "* Insert for employee 10: customers_assigned=actual count, performance_score=4.2\n\n"
             "### 6. DELETE: Remove Robert King (EmployeeId=7)\n"
             "* First reassign employees who report to 7 → report to 7's manager (ReportsTo of 7)\n"
             "* Reassign customers with SupportRepId=7 → to 7's manager\n"
             "* Then DELETE employee 7\n\n"
             "### 7. UPDATE: Promote Laura Callahan\n"
             "* EmployeeId=8: Title → 'Senior IT Specialist'\n"
             "* Add `salary` column (REAL) to Employee table\n"
             "* Laura (8) salary=75000.00, all others salary=50000.00")
    c["verify"] = [
        {"sql": 'SELECT COUNT(*) FROM "Employee"', "expected": [[9]], "order_matters": False},
        {"sql": 'SELECT "Title" FROM "Employee" WHERE "EmployeeId"=1', "expected": [["CEO"]], "order_matters": False},
        {"sql": 'SELECT "Phone" FROM "Employee" WHERE "EmployeeId"=2', "expected": [["+1 (403) 555-9999"]], "order_matters": False},
        {"sql": 'SELECT COUNT(*) FROM "Employee" WHERE "Title"=\'IT Specialist\'', "expected": [[0]], "order_matters": False},
        {"sql": 'SELECT "ReportsTo" FROM "Employee" WHERE "EmployeeId"=9', "expected": [[1]], "order_matters": False},
        {"sql": 'SELECT COUNT(*) FROM "Customer" WHERE "SupportRepId"=9', "expected": [[3]], "order_matters": False},
        {"sql": 'SELECT employee_id, customers_assigned FROM employee_performance ORDER BY employee_id',
         "expected": [[9, 3], [10, 3]], "order_matters": True},
        {"sql": 'SELECT "Title","salary" FROM "Employee" WHERE "EmployeeId"=8',
         "expected": [["Senior IT Specialist", 75000.0]], "order_matters": False},
    ]
    write_case(c)

    # ===== 017: sales_and_music_charts =====
    monthly_count = q("chinook", """
        SELECT COUNT(DISTINCT substr("InvoiceDate",1,7)) FROM "Invoice"
    """)[0][0]
    c = base("mcpmark_postgres_017", "chinook",
             ["reporting and analytics", "rankings", "difficulty:L3", "database:chinook"],
             "Create a monthly sales dashboard and top music charts.\n\n"
             "### 1. `monthly_sales_summary` table\n"
             "* `year_month` (varchar — format 'YYYY-MM')\n"
             "* `total_invoices` (integer), `total_revenue` (real), `total_tracks_sold` (integer)\n"
             "* `average_invoice_value` (real), `unique_customers` (integer)\n"
             "* Use substr(InvoiceDate,1,7) or strftime for year-month extraction\n"
             "* total_tracks_sold = SUM(Quantity) from InvoiceLine\n\n"
             "### 2. `top_music_charts` table\n"
             "* `chart_type` (varchar — 'top_tracks','top_albums','top_artists')\n"
             "* `rank_position` (integer 1-10), `item_id` (integer), `item_name` (varchar), `total_revenue` (real)\n"
             "* Top Tracks: rank by total Quantity sold, tiebreak by name ASC\n"
             "* Top Albums: rank by total revenue from tracks, tiebreak by album name ASC\n"
             "* Top Artists: rank by total revenue from all tracks, tiebreak by artist name ASC\n"
             "* Include only top 10 for each chart type")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM monthly_sales_summary", "expected": [[monthly_count]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM top_music_charts", "expected": [[30]], "order_matters": False},
        {"sql": "SELECT COUNT(DISTINCT chart_type) FROM top_music_charts", "expected": [[3]], "order_matters": False},
    ]
    write_case(c)

    # ===== 018: customer_analytics_optimization =====
    c = base("mcpmark_postgres_018", "dvdrental",
             ["indexing", "query optimization", "difficulty:L3", "database:dvdrental"],
             "Optimize a slow customer analytics query in the DVD rental database.\n\n"
             "## Background\n\n"
             "The BI team's critical customer analytics query is timing out. Analyze and optimize it.\n\n"
             "## The Slow Query\n\n"
             "```sql\n"
             "SELECT\n"
             "    c.customer_id, c.first_name, c.last_name, c.email,\n"
             "    COUNT(DISTINCT p.payment_id) as total_payments,\n"
             "    SUM(p.amount) as total_spent,\n"
             "    AVG(p.amount) as avg_payment,\n"
             "    MAX(p.payment_date) as last_payment,\n"
             "    MIN(p.payment_date) as first_payment\n"
             "FROM customer c\n"
             "JOIN payment p ON c.customer_id = p.customer_id\n"
             "WHERE c.active = 1\n"
             "GROUP BY c.customer_id, c.first_name, c.last_name, c.email\n"
             "HAVING COUNT(p.payment_id) >= 10\n"
             "ORDER BY total_spent DESC, total_payments DESC;\n"
             "```\n\n"
             "## Your Task\n\n"
             "1. Analyze the query to identify performance bottlenecks\n"
             "2. Create appropriate indexes to optimize the query\n"
             "3. The key optimization needed is an index on `payment(customer_id)` to speed up the JOIN\n\n"
             "## Requirements\n"
             "- Create an index on the `payment` table's `customer_id` column\n"
             "- You may create additional indexes if needed")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='payment' AND sql LIKE '%customer_id%'",
         "expected": [[1]], "order_matters": False},
    ]
    write_case(c)

    # ===== 019: film_inventory_management =====
    c = base("mcpmark_postgres_019", "dvdrental",
             ["data manipulation", "inventory management", "difficulty:L3", "database:dvdrental"],
             "Manage film inventory operations in the DVD rental database.\n\n"
             "Complete these operations in sequence:\n\n"
             "### 1. Add New Films\n"
             "* Title='Data Science Adventures', Description='A thrilling journey through machine learning algorithms', release_year=2024, language_id=1, rental_duration=5, rental_rate=3.99, length=120, replacement_cost=15.99, rating='PG-13'\n"
             "* Title='Cloud Computing Chronicles', Description='Exploring the world of distributed systems', release_year=2024, language_id=1, rental_duration=7, rental_rate=4.99, length=135, replacement_cost=18.99, rating='PG'\n\n"
             "### 2. Add Inventory Records\n"
             "For each new film: 3 inventory records for store_id=1, 2 for store_id=2\n\n"
             "### 3. Update Rental Rates\n"
             "Increase rental_rate by 10% (multiply by 1.1) for ALL films with rating='PG-13'\n\n"
             "### 4. Create `available_films` Table\n"
             "* film_id (INTEGER PRIMARY KEY), title (TEXT NOT NULL), rental_rate (REAL NOT NULL), length (INTEGER)\n"
             "* Include films with: rental_rate between 3.00 and 5.00, length > 100, at least 1 inventory record in store_id=1\n\n"
             "### 5. Clean Up Inventory\n"
             "Delete inventory records for films with ALL: replacement_cost > 25.00, rental_rate < 1.00, and no rental history\n\n"
             "### 6. Create `film_inventory_summary` Table\n"
             "* title (TEXT NOT NULL), rental_rate (REAL NOT NULL), total_inventory (INTEGER), store1_count (INTEGER), store2_count (INTEGER)\n"
             "* Only films with at least 1 inventory record\n"
             "* Sort by total_inventory DESC, then title ASC")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM film WHERE title IN ('Data Science Adventures','Cloud Computing Chronicles')",
         "expected": [[2]], "order_matters": False},
        {"sql": "SELECT title, ROUND(rental_rate,2) FROM film WHERE title='Data Science Adventures'",
         "expected": [["Data Science Adventures", 4.39]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM available_films WHERE rental_rate BETWEEN 3.0 AND 5.0 AND length > 100",
         "expected": [[q("dvdrental", """
             SELECT COUNT(DISTINCT f.film_id) FROM film f
             JOIN inventory i ON f.film_id=i.film_id AND i.store_id=1
             WHERE ROUND(f.rental_rate*CASE WHEN f.rating='PG-13' THEN 1.1 ELSE 1.0 END,2) BETWEEN 3.0 AND 5.0
             AND f.length > 100
         """)[0][0]]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM film_inventory_summary", "expected": [[q("dvdrental",
             "SELECT COUNT(DISTINCT f.film_id) FROM film f JOIN inventory i ON f.film_id=i.film_id")[0][0] + 2]],
         "order_matters": False},
    ]
    write_case(c)

    # ===== 020: participant_report_optimization =====
    report_rows = q("sports", """
        SELECT pe.participant_id, COUNT(pe.event_id) as event_count
        FROM participants_events pe WHERE pe.participant_id <= 50
        GROUP BY pe.participant_id ORDER BY pe.participant_id
    """)
    n_participants = len(report_rows)
    c = base("mcpmark_postgres_020", "sports",
             ["query optimization", "indexing", "difficulty:L3", "database:sports"],
             "Create a performance report and optimize a slow query.\n\n"
             "### 1. Create `participant_performance_report` table\n"
             "* report_id (INTEGER PRIMARY KEY), participant_id (INTEGER NOT NULL), event_count (INTEGER), stat_count (INTEGER), stat_type_count (INTEGER), last_event_date (TEXT), created_at (TEXT DEFAULT (datetime('now')))\n"
             "* Add CHECK constraint: participant_id > 0\n\n"
             "### 2. Create Optimization Indexes\n"
             "* Index on participants_events(participant_id)\n"
             "* Composite index on stats(stat_holder_type, stat_holder_id)\n\n"
             "### 3. Populate Report Table\n"
             "Execute this query and insert results into the report table:\n"
             "```sql\n"
             "SELECT\n"
             "    pe.participant_id,\n"
             "    COUNT(pe.event_id) as event_count,\n"
             "    (SELECT COUNT(*) FROM stats s WHERE s.stat_holder_id = pe.participant_id AND s.stat_holder_type = 'persons') as stat_count,\n"
             "    (SELECT COUNT(DISTINCT s.stat_repository_type) FROM stats s WHERE s.stat_holder_id = pe.participant_id AND s.stat_holder_type = 'persons') as stat_type_count,\n"
             "    (SELECT MAX(e.start_date_time) FROM events e JOIN participants_events pe2 ON e.id = pe2.event_id WHERE pe2.participant_id = pe.participant_id) as last_event_date\n"
             "FROM participants_events pe\n"
             "WHERE pe.participant_id <= 50\n"
             "GROUP BY pe.participant_id\n"
             "ORDER BY pe.participant_id;\n"
             "```")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM participant_performance_report", "expected": [[n_participants]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='participants_events' AND sql LIKE '%participant_id%'",
         "expected": [[1]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='stats' AND sql LIKE '%stat_holder_type%' AND sql LIKE '%stat_holder_id%'",
         "expected": [[1]], "order_matters": False},
    ]
    write_case(c)

    # ===== 021: team_roster_management =====
    # Count players with offensive stats and >=10 games
    n_eval_players = q("sports", """
        SELECT COUNT(DISTINCT s.stat_holder_id) FROM stats s
        JOIN baseball_offensive_stats bos ON bos.id=s.stat_repository_id
        JOIN person_event_metadata pem ON pem.person_id=s.stat_holder_id
        WHERE s.stat_holder_type='persons' AND s.stat_repository_type='baseball_offensive_stats'
        AND s.context='season-regular'
        GROUP BY s.stat_holder_id
        HAVING COUNT(DISTINCT pem.event_id) >= 10
    """)
    player_eval_count = len(n_eval_players)
    c = base("mcpmark_postgres_021", "sports",
             ["data manipulation", "roster management", "difficulty:L3", "database:sports"],
             "Manage team rosters with performance tracking and injury analysis.\n\n"
             "### 1. Create `player_evaluation` table\n"
             "* performance_id (INTEGER PRIMARY KEY), person_id (INTEGER NOT NULL), batting_avg (REAL), home_runs (INTEGER), rbis (INTEGER), games_played (INTEGER), performance_score (REAL), evaluation_date (TEXT)\n"
             "* CHECK constraint: batting_avg BETWEEN 0 AND 1\n\n"
             "### 2. Load Player Statistics\n"
             "Insert from baseball data:\n"
             "* batting_avg = hits/at_bats (0 if at_bats=0)\n"
             "* Sum home_runs, rbi from baseball_offensive_stats (context='season-regular')\n"
             "* games_played = COUNT(DISTINCT event_id) from person_event_metadata\n"
             "* performance_score = (batting_avg * 1000) + (home_runs * 5) + (rbis * 2)\n"
             "* Only players with games_played >= 10\n"
             "* evaluation_date = '2024-01-01'\n\n"
             "### 3. Create `player_injury_status` table\n"
             "* status_id (INTEGER PRIMARY KEY), person_id (INTEGER UNIQUE NOT NULL), injury_count (INTEGER DEFAULT 0), last_injury_date (TEXT), current_status (TEXT CHECK(current_status IN ('healthy','injured','recovering')))\n"
             "* Include all players from player_evaluation\n"
             "* Count injuries from injury_phases\n"
             "* current_status: 'injured' if any injury has no end_date_time, else 'healthy'\n\n"
             "### 4. Adjust Scores Based on Health\n"
             "* Reduce performance_score by 20% for 'injured' players\n"
             "* Reduce performance_score by 10% for players with injury_count > 2\n"
             "* Minimum score = 0\n\n"
             "### 5. Create `team_performance_summary` table\n"
             "* summary_id (INTEGER PRIMARY KEY), metric_name (TEXT UNIQUE), metric_value (REAL)\n"
             "* Insert: 'total_players', 'avg_batting_average', 'total_home_runs', 'avg_performance_score', 'injured_player_count', 'healthy_player_count'")
    c["verify"] = [
        {"sql": "SELECT COUNT(*) FROM player_evaluation WHERE games_played >= 10",
         "expected": [[player_eval_count]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM player_injury_status", "expected": [[player_eval_count]], "order_matters": False},
        {"sql": "SELECT COUNT(*) FROM team_performance_summary", "expected": [[6]], "order_matters": False},
        {"sql": "SELECT metric_name FROM team_performance_summary ORDER BY metric_name",
         "expected": [["avg_batting_average"],["avg_performance_score"],["healthy_player_count"],["injured_player_count"],["total_home_runs"],["total_players"]],
         "order_matters": True},
    ]
    write_case(c)

    print(f"\n14 cases written (mcpmark_postgres_008 through mcpmark_postgres_021)")


if __name__ == "__main__":
    main()
