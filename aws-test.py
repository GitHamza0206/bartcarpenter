import boto3
import pandas as pd
from io import StringIO

session = boto3.Session(
    aws_access_key_id='AKIATCPYKRBDNS5OE7XH',
    aws_secret_access_key='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',
    region_name='us-east-1'
)

# AWS details
bucket_name = 'receiptscanners3'
file_name = 'Bank Upload CSV QBO (2).csv'
s3 = session.client('s3')

print(s3.get_object(Bucket=bucket_name,Key=file_name))

# Function to download file from S3
def download_file_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

# Function to upload file to S3
def upload_file_to_s3(bucket, key, data):
    s3.put_object(Bucket=bucket, Key=key, Body=data)

# Download the file
#csv_data = download_file_from_s3(bucket_name, file_name)
csv_data = download_file_from_s3(bucket_name, file_name)

# Load into pandas, edit as needed
df = pd.read_csv(StringIO(csv_data))
# Perform your edits here, e.g., df['new_column'] = df['existing_column'] * 2
# df = pd.DataFrame(columns=["Date", "Amount","Vendor","Expense Type"])

# #row = [{"Date":1, "Amount":0,"Vendor":0,"Expense Type":1}] 
# #df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
# # Save the DataFrame back to CSV
# csv_buffer = StringIO()
# df.to_csv(csv_buffer, index=False)
# csv_buffer.seek(0)
print(df.head())
# Upload the edited file

#upload_file_to_s3(bucket_name, file_name, csv_buffer.getvalue())