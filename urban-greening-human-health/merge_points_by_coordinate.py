import pandas as pd
import os

def merge_by_coordinates():
    # Table file path (backslashes already replaced with forward slashes)
    file_path = "C:/Users/DELL/Desktop/数据/数据/匹配后的样本点数据.csv"

    try:
        # Check that the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Read CSV data
        print("Reading tabular data...")
        df = pd.read_csv(file_path)

        # Check that the required columns exist
        required_columns = ["经度", "纬度", "收缩压", "舒张压", "平均动脉压", "血糖", "年龄", "收入"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the table")

        # Columns to be averaged
        avg_columns = ["收缩压", "舒张压", "平均动脉压", "血糖", "年龄", "收入"]

        # Other columns (keep the first row's value)
        other_columns = [col for col in df.columns if col not in avg_columns and col not in ["经度", "纬度"]]

        # Define aggregation: average the columns to be averaged, take first value for the others
        agg_functions = {}
        for col in avg_columns:
            agg_functions[col] = 'mean'
        for col in other_columns + ["经度", "纬度"]:
            agg_functions[col] = 'first'

        # Group by longitude/latitude and apply the aggregation
        print("Merging data by longitude/latitude...")
        merged_df = df.groupby(["经度", "纬度"], as_index=False).agg(agg_functions)

        # Save the result to a new CSV file
        output_dir = os.path.dirname(file_path)
        output_path = os.path.join(output_dir, "merged_sample_points.csv")
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Merge complete! Result saved to: {output_path}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    merge_by_coordinates()
