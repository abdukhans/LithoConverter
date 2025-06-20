import csv
import sqlite3
from ast import literal_eval

class CSVProcessor:
    def __init__(self):
        self.db_file = 'similarKeywords.db'
        self.conn = None
        self.combine_num = 0
    
    def create_database(self):
        """Create the SQLite database and table structure"""
        self.conn = sqlite3.connect(self.db_file)
        cursor = self.conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS similarKeywords (
            Attribute TEXT PRIMARY KEY,
            Frequency INTEGER,
            Similar_Words TEXT,
            rm BOOLEAN DEFAULT FALSE,
            combine INTEGER DEFAULT 0
        )
        ''')
        
        self.conn.commit()
    
    def import_original_csv(self, csvFile):
        """Import data from original.csv into the database"""
        cursor = self.conn.cursor()
        # 'original.csv'
        with open(csvFile, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute('''
                INSERT OR REPLACE INTO similarKeywords (Attribute, Frequency, Similar_Words)
                VALUES (?, ?, ?)
                ''', (row['Attribute'], int(row['Frequency']), row['Similar_Words']))
        
        self.conn.commit()
    def mark_single_letter_attributes(self):
        """Mark all single-letter Attribute records for removal"""
        cursor = self.conn.cursor()
        cursor.execute('''
        UPDATE similarKeywords
        SET rm = TRUE
        WHERE length(trim(Attribute)) = 1
        ''')
        count = cursor.rowcount
        self.conn.commit()
        print(f"Marked {count} single-letter Attribute records for removal")

    def process_task_file(self,csvFile):
        """Process the task.csv file and update the database accordingly"""
        cursor = self.conn.cursor()
        # 'task.csv'
        with open(csvFile, 'r') as f:
            # Read with stripped whitespace and case-insensitive column matching
            reader = csv.DictReader(f, skipinitialspace=True)
            
            # Normalize column names by stripping whitespace and making lowercase
            fieldnames = [name.strip().lower() for name in reader.fieldnames]
            reader.fieldnames = fieldnames
            
            for row in reader:
                task_type = row['task'].strip().lower()
                words = row['word'].strip('"').strip()
                
                if task_type == 'rm':
                    # Process removal task
                    word = words
                    cursor.execute('''
                    UPDATE similarKeywords
                    SET rm = TRUE
                    WHERE Attribute = ?
                    ''', (word,))
                    print(f"Marked '{word}' for removal")
                    
                elif task_type == 'combine':
                    # Process combine task
                    self.combine_num += 1
                    word_list = [w.strip() for w in words.split(',')]
                    
                    for word in word_list:
                        cursor.execute('''
                        UPDATE similarKeywords
                        SET combine = ?
                        WHERE Attribute = ?
                        ''', (self.combine_num, word))
                        print(f"Added '{word}' to combine group {self.combine_num}")
        
        self.conn.commit()
    
    def generate_new_csv(self,csvFile):
        """Generate the new.csv file based on the processed data"""
        cursor = self.conn.cursor()
        #'new.csv'
        with open(csvFile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Attribute', 'Frequency', 'Similar_Words'])
            
            # Write records with combine==0 and rm==FALSE
            cursor.execute('''
            SELECT Attribute, Frequency, Similar_Words
            FROM similarKeywords
            WHERE combine = 0 AND rm = FALSE
            ''')
            
            for row in cursor.fetchall():
                writer.writerow(row)
            
            # Process and write combined records
            for i in range(1, self.combine_num + 1):
                cursor.execute('''
                SELECT Attribute, Frequency, Similar_Words
                FROM similarKeywords
                WHERE combine = ?
                ORDER BY Attribute
                ''', (i,))
                
                records = cursor.fetchall()
                if not records:
                    continue
                    
                # Combine attributes
                main_attribute = records[0][0]
                total_frequency = sum(record[1] for record in records)
                
                # Combine similar words
                similar_words = []
                for record in records:
                    try:
                        # Safely evaluate the string as a Python literal
                        words_list = literal_eval(record[2])
                        if isinstance(words_list, list):
                            similar_words.extend(words_list)
                    except (ValueError, SyntaxError):
                        # Handle malformed Similar_Words data
                        pass
                
                # Format the combined similar words
                combined_similar_words = str(similar_words)
                
                # Write the combined record
                writer.writerow([main_attribute, total_frequency, combined_similar_words])
    
    def close_connection(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
    
    def run(self):
        """Run the complete processing pipeline"""
        try:
            self.create_database()
            self.import_original_csv('similar_keywords_freq.csv') #orignal csv file name
            self.mark_single_letter_attributes()
            self.process_task_file('task.csv') # task is a csv file
            self.generate_new_csv('new_keywords.csv') # please change the csv file name
            print("Processing complete. Output written to new csv file")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.close_connection()

if __name__ == '__main__':
    processor = CSVProcessor()
    processor.run()

# task.csv demo
# Task, Word
# rm, "rock"
# combine, "rhyolite, schist"