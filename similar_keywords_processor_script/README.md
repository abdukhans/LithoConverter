# READMEN


## How to use

1. Delete files
- similarKeywords.db
- new.csv

2. setting task

change setting at the task.csv

```csv
Task, Word
rm, "rock" # rm rock 
combine, "rhyolite, schist" # combine schist to rhyolite
```

3. run script
```
python csv_processor.py 
```
output log like this

```bash
Marked 2 single-letter Attribute records for removal
Marked 'rock' for removal
Added 'rhyolite' to combine group 1
Added 'schist' to combine group 1
Processing complete. Output written to new.csv
```

## Python Script Requirements

I need to write a Python script that performs the following operations on a CSV file:

* Read data from a CSV file
* Delete specific rows
* Merge rows based on certain rules
* Write the processed data to a new CSV file

### 📋 Detailed Requirements:

#### 1. Create a text-based database (named `similarKeywords`)

* The database contains a **single table**: `similarKeywords`
* Table fields:

  * `Attribute`: text
  * `Frequency`: number
  * `Similar_Words`: text
  * `rm`: boolean, default `false`
  * `combine`: number, default `0`

---

#### 2. Read a file named `original.csv` and load its content into the `similarKeywords` table.

**Sample content of original.csv**:

```csv
Attribute,Frequency,Similar_Words
igneous,188361,"[('igneous', 188361)]"
sedimentary,93604,"[('sedimentary', 93604)]"
metamorphic,69654,"[('metamorphic', 69654)]"
```

---

#### 3. Read another file named `task.csv` with the following structure:

```csv
task,word
rm,"rock"
combine,"rhyolite, schist"
```

Process each task as follows:

* **Task 1: rm "rock"**

  * Look up the row where `Attribute = "rock"`
  * Set `rm = true` for that row

* **Task 2: combine "rhyolite, schist"**

  * Keep a counter `combineNum`, starting from `1` and incrementing by 1 for each combine task
  * For each word in the `word` field (e.g., "rhyolite" and "schist"), find the corresponding rows where `Attribute` matches, and set `combine = combineNum`

---

#### 4. Create a new file `new.csv` with the following header:

```csv
Attribute,Frequency,Similar_Words
```

---

#### 5. Write selected records from the `similarKeywords` table to `new.csv`:

**Step 1: Write unmarked rows**

* Filter rows where `combine == 0` **and** `rm == false`
* Write them to `new.csv` as-is

**Step 2: Write merged rows**

* For each unique value of `combine` (from `1` to `combineNum`):

  * Filter all rows with the same `combine` value
  * Merge them into one new row:

    * `Attribute`: use the `Attribute` value from the **first** row
    * `Frequency`: sum of `Frequency` of all combined rows
    * `Similar_Words`: merge all `Similar_Words` lists together into a single list

      * Example: `"[('sedimentary', 93604)]"` + `"[('metamorphic', 69654)]"` → `"[('sedimentary', 93604), ('metamorphic', 69654)]"`
  * Write the merged result to `new.csv`
