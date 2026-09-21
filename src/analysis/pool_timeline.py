import sqlite3
import pandas as pd
from decimal import Decimal, getcontext
import math
import argparse

"""
Merge the liquidity and fee_and_price tables to create a timeline table that contains events of 
'delta_liquidity' and 'swap' in the increasing order of datetime values.

"""
CONSOLIDATED_INSERT_TIMELINE_QUERY = """
    CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            block_ts_utc TEXT,
            event_type TEXT,
            pool_id TEXT,
            sqrt_price_x96 TEXT,
            delta_liquidity TEXT,
            tick_lower INTEGER,
            tick_upper INTEGER,
            amount0 TEXT,
            amount1 TEXT,
            liquidity TEXT,
            tx_hash TEXT,
            block_number TEXT
        );

    INSERT INTO {table_name} (block_ts_utc, event_type, pool_id, sqrt_price_x96, delta_liquidity, tick_lower, tick_upper,
                        amount0, amount1, liquidity, tx_hash, block_number)
        SELECT 
            block_ts_utc AS block_ts_utc, 
            'mod_liquidity' AS event_type, 
            pool_id AS pool_id, 
            sqrt_price_x96 AS sqrt_price_x96, 
            delta_liquidity AS delta_liquidity, 
            tick_lower AS tick_lower, 
            tick_upper AS tick_upper, 
            NULL AS amount0, 
            NULL AS amount1, 
            NULL AS liquidity, 
            tx_hash AS tx_hash, 
            block_number AS block_number
        FROM curr_db.liquidity_positions

        UNION ALL

        SELECT 
            block_ts_utc AS block_ts_utc, 
            'swap' AS event_type, 
            pool_id AS pool_id,
            sqrt_price_x96 AS sqrt_price_x96,
            NULL AS delta_liquidity, 
            NULL AS tick_lower,
            NULL AS tick_upper, 
            amount0 AS amount0,
            amount1 AS amount1, 
            liquidity AS liquidity, 
            tx_hash AS tx_hash, 
            block_number AS block_number
        FROM curr_db.fee_and_price_history

        ORDER BY block_ts_utc;
    """

# Create timeline table in a file itself
def merge_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create the unified timeline table
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS timeline (
            id INTEGER PRIMARY KEY,
            block_ts_utc TEXT,
            event_type TEXT,
            pool_id TEXT,
            sqrt_price_x96 TEXT,
            delta_liquidity TEXT,
            tick_lower INTEGER,
            tick_upper INTEGER,
            amount0 TEXT,
            amount1 TEXT,
            liquidity TEXT,
            tx_hash TEXT,
            block_number TEXT
        );

        DELETE FROM timeline;  -- Clear previous data if re-running

        INSERT INTO timeline (block_ts_utc, event_type, pool_id, sqrt_price_x96, delta_liquidity, tick_lower, tick_upper,
                        amount0, amount1, liquidity, tx_hash, block_number)
        SELECT block_ts_utc, 'mod_liquidity', pool_id, sqrt_price_x96, delta_liquidity,  tick_lower, tick_upper, 
            NULL, NULL, NULL, tx_hash, block_number
        FROM liquidity_positions

        UNION ALL

        SELECT block_ts_utc, 'swap', pool_id, sqrt_price_x96, NULL, NULL, NULL, 
            amount0, amount1, liquidity, tx_hash, block_number
        FROM fee_and_price_history

        ORDER BY block_ts_utc;
    """)
    conn.commit()

def combine_consecutive_mod_liquidity():
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS timeline_aggregated (
            block_ts_utc TEXT,
            event_type TEXT,
            pool_id TEXT,
            sqrt_price_x96 TEXT,
            delta_liquidity TEXT,
            tick_lower INTEGER,
            tick_upper INTEGER,
            amount0 TEXT,
            amount1 TEXT,
            liquidity TEXT,
            tx_hash TEXT,
            block_number TEXT
        );

        DELETE FROM timeline_aggregated;

        WITH numbered AS (
            SELECT 
                block_ts_utc,
                event_type,
                pool_id,
                sqrt_price_x96,
                delta_liquidity,
                tick_lower,
                tick_upper,
                amount0,
                amount1,
                liquidity,
                tx_hash,
                block_number,
                ROW_NUMBER() OVER (ORDER BY block_ts_utc) 
                - ROW_NUMBER() OVER (PARTITION BY event_type ORDER BY block_ts_utc) AS grp
            FROM timeline
        ),
        consecutive_groups AS (
            SELECT 
                MAX(block_ts_utc) AS datetime,
                SUM(delta_liquidity) AS total_delta_liquidity,
                COUNT(*) AS event_count
            FROM numbered
            WHERE event_type = 'mod_liquidity'
            GROUP BY grp
        )
        INSERT INTO timeline_aggregated (datetime, total_quantity, event_count)
        SELECT datetime, total_quantity, event_count
        FROM consecutive_groups
        ORDER BY datetime;
    """)

def check_split(db_path, split_type):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count records
    cursor.execute(f"SELECT COUNT(id) FROM timeline where split = {split_type}")
    total_size = cursor.fetchone()[0]
    conn.close()
    print("Total count: ", total_size)

def add_train_val_split_col(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count records
    cursor.execute("SELECT COUNT(id) FROM timeline")
    total_size = cursor.fetchone()[0]
    val_size = int(0.2 * total_size)          # e.g., 20% for validation
    train_size = total_size - val_size

    # Add the new 'split' column
    cursor.execute("ALTER TABLE timeline ADD COLUMN split INTEGER")

    # Initialize all rows to 0
    cursor.execute("UPDATE timeline SET split = 0")

    # Randomly assign val_sizerows to 1
    print(f"Setting {val_size} validation data")
    #cursor.execute("""
    #    UPDATE timeline 
    #    SET split = 1
    #    WHERE rowid IN (
    #        SELECT id 
    #        FROM timeline 
    #        ORDER BY RANDOM() 
    #        LIMIT ?
    #    )
    #""", (val_size,))

    cursor.execute("""
       UPDATE timeline
       SET split = 1
       WHERE block_ts_utc > '2026-02-23 00:00:00';
    """)

    conn.commit()
    conn.close()
    print("Done!")

def add_train_val_split_col2(db_path):
    query = """
    WITH OrderedData AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY id) - 1 AS row_num
    FROM timeline
    )
    UPDATE timeline
    SET split = CASE 
        --- Suppose dataset has N records. Find a,b: 0.2N = a x b. If N is odd number, take a multiple of 10 close to N.
        --- Then OrderedData.row_num % (b/0.2) < b
        -- 97 chunks of 2442 records = 236874 records total (20%)
        WHEN (OrderedData.row_num % 5050) < 1000 THEN 1 
        ELSE 0 
    END
    FROM OrderedData
    WHERE timeline.id = OrderedData.id;
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    conn.close()
    print("Done!")

def remove_none_price(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
       UPDATE timeline
       SET sqrt_price_x96 = '0.000000000001' 
       WHERE sqrt_price_x96 == '000000000001';
    """)

    conn.commit()
    conn.close()
    print("Done!")
    

def consolidate_timelines(db_files, output_path):
    table_name = 'timeline'

    for db_file in db_files:
        try:
            # Connect to the database file
            new_conn = sqlite3.connect(output_path)
            cursor = new_conn.cursor()
            print(f'Attaching to {db_file}')
            cursor.execute(f"ATTACH DATABASE '{db_file}' AS curr_db")
            query_string = CONSOLIDATED_INSERT_TIMELINE_QUERY.format(table_name=table_name)
            #print(query_string)
            cursor.executescript(query_string)
            new_conn.commit() 
        except Exception as e:
            print(f"Error processing {db_file}: {e}")
        finally:
            if 'new_conn' in locals():
                new_conn.close()
                print("Done!")

def main(output_path: str, op: str = 'noop', tag: str = 'wbtc_usdc'):
    TAG = tag
    db_files = [
        f"../curation/sqldb/{TAG}/" + f"swaps_univ4_{TAG}_21191360_to_21835359.db",
        f"../curation/sqldb/{TAG}/" + f"swaps_univ4_{TAG}_21835360_to_22480359.db",
        f"../curation/sqldb/{TAG}/" + f"swaps_univ4_{TAG}_22480360_to_23120359.db",
        f"../curation/sqldb/{TAG}/" + f"swaps_univ4_{TAG}_23120360_to_23760359.db",
        f"../curation/sqldb/{TAG}/" + f"swaps_univ4_{TAG}_23760360_to_24430359.db",
        f"../curation/sqldb/{TAG}/" + f"swaps_univ4_{TAG}_24430360_to_25044360.db",
    ]

    if op == 'consolidate_timelines':
        consolidate_timelines(db_files, output_path)
    elif op == 'remove_none_price':
        remove_none_price(output_path)
    elif op == 'train_val_split':
        add_train_val_split_col2(output_path)
    elif op == 'check_split':
        check_split(db_path=output_path, split_type=1)
    else:
        print('No operation done!')


if __name__ == "__main__":
    
    tag = 'wbtc_usdc'
    output_path = f'sqldb/{tag}/{tag}_univ4_timeline.db'
    
    parser = argparse.ArgumentParser(description="CLI options")
    parser.add_argument("-p", "--operation", type=str, default='noop', help="Operation to perform")
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Pass the arguments directly into your main function
    main(output_path=output_path, op=args.operation, tag=tag)


