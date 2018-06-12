import pyspark
from math import sin, cos, sqrt, atan2, radians

# Function to calculate the distance of gps coordinates
def compute_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6371.0 
    
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance*1000

def generate_metadata_pandas(spark):
    metadata = spark.read.text('data/metadata/BFKOORD_GEO')

    # Splitting and adding new columns to the spark dataframe
    split_col = pyspark.sql.functions.split(metadata['value'], " % ")
    split_left = pyspark.sql.functions.split(split_col.getItem(0), " +")
    metadata = metadata.withColumn('longitude', split_left.getItem(2))
    metadata = metadata.withColumn('latitude', split_left.getItem(1))
    metadata = metadata.withColumn('stop', split_col.getItem(1))
    metadata = metadata.drop('value')

    # Save in a pandas DataFrame
    metadataPandas = metadata.toPandas()
    zurich_coord = metadataPandas[metadataPandas['stop'] == 'Zürich HB']

    # keep only stations inside 10km
    mask = metadataPandas.apply(lambda x: compute_distance(
	    float(zurich_coord['latitude']), 
	    float(zurich_coord['longitude']),
	    float(x['latitude']), 
	    float(x['longitude'])) <= 10000, axis=1)
    metadataPandas = metadataPandas[mask]

    return metadataPandas
