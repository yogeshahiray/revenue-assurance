import os
from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath
import lzma
import shutil
from kfp import kubernetes

@dsl.component(base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2024a-20240523")
def get_data(train_data_output_path: OutputPath()):
    import os
    import boto3
    import botocore
    import urllib.request

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    s3_key = os.environ.get("S3_DATA_KEY")
    print("starting download...")
    print("downloading training data")
    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)
    bucket.download_file(s3_key, train_data_output_path)

    print(f"Data file downloaded to {train_data_output_path}")

    # url = "https://github.com/tme-osx/TME-AIX/blob/main/revenueassurance/data/telecom_revass_data.csv.xz"
    # urllib.request.urlretrieve(url, train_data_output_path)

    # # Extract the .xz file
    # with lzma.open(train_data_output_path, 'rb') as f_in:
    #     with open("telecom_revass_data.csv", 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    print("train data downloaded")
    # print("downloading validation data")
    # url = "https://raw.githubusercontent.com/cfchase/fraud-detection/main/data/validate.csv"
    # urllib.request.urlretrieve(url, validate_data_output_path)
    # print("validation data downloaded")


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2024a-20240523",
    packages_to_install=["onnx", "onnxruntime", "tf2onnx"],
)
def train_model(train_data_input_path: InputPath(), model_output_path: OutputPath()):
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight
    import tf2onnx
    import onnx
    import pickle
    from pathlib import Path

    # fields:
    # * ** Call_Duration **
    # * ** Data_Usage **
    # * ** SMS_Count **
    # * ** Roaming_Indicator **
    # * ** MobileWallet_Use **
    # * ** Cost **
    # * ** Cellular_Location_Distance **
    # * ** Personal_Pin_Used **
    # * ** Avg_Call_Duration **
    # * ** Avg_Data_Usage **
    # * ** Avg_Cost **
    # * ** fraud ** - If the transaction is fraudulent.

    feature_indexes = [
        0,  # Call_Duration
        1,  # Data_Usage
        2,  # SMS_Count
        3,  # Roaming_Indicator
        4,  # MobileWallet_Use
        6,  # Cost
        7,  # Cellular_Location_Distance
        8,  # Personal_Pin_Used
        9,  # Avg_Call_Duration
        10,  # Avg_Data_Usage
        11  # Avg_Cost
    ]

    label_indexes = [
        12  # fraud
    ]

    df = pd.read_csv(train_data_input_path)
    X = df.iloc[:, feature_indexes].values
    y = df.iloc[:, label_indexes].values

    print(df.info)

    print(df.head)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # X_val = pd.read_csv(validate_data_input_path)
    # y_val = X_val.iloc[:, label_indexes]
    # X_val = X_val.iloc[:, feature_indexes]

    # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier for the model to learn than random (and potentially large) values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    Path("artifact").mkdir(parents=True, exist_ok=True)
    with open("artifact/scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # here, Build the model, the model we build here is a simple fully connected deep neural network, containing 3 hidden layers and one output layer.

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=len(feature_indexes)))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model and get performance

    epochs = 2
    history = model.fit(X_train, y_train, epochs=epochs,
                        verbose=True, class_weight=class_weights)

    # Save the model as ONNX for easy use of ModelMesh
    model_proto, _ = tf2onnx.convert.from_keras(model)
    print(model_output_path)
    onnx.save(model_proto, model_output_path)
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2024a-20240523",
    packages_to_install=["boto3", "botocore"]
)
def upload_model(input_model_path: InputPath()):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    s3_key = os.environ.get("S3_KEY")

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading {s3_key}")
    bucket.upload_file(input_model_path, s3_key)


@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline():
    get_data_task = get_data()
    get_data_task.set_env_variable(name="S3_DATA_KEY", value="train-data/telecom_revass_data.csv")
    kubernetes.use_secret_as_env(
        task=get_data_task,
        secret_name='aws-connection-rafm-storage',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })

    train_data_csv_file = get_data_task.outputs["train_data_output_path"]
    # validate_data_csv_file = get_data_task.outputs["validate_data_output_path"]

    train_model_task = train_model(train_data_input_path=train_data_csv_file)
                                   # validate_data_input_path=validate_data_csv_file)
    onnx_file = train_model_task.outputs["model_output_path"]

    upload_model_task = upload_model(input_model_path=onnx_file)

    upload_model_task.set_env_variable(name="S3_KEY", value="models/rafm/1/rafmmodel.onnx")

    kubernetes.use_secret_as_env(
        task=upload_model_task,
        secret_name='aws-connection-rafm-storage',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )
