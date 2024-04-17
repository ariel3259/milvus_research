from pymilvus import MilvusClient, DataType
from openai import OpenAI
import json

##CREATE Milvus and openai clients
milvus_client = MilvusClient()
openai_client = OpenAI()
collection_name = "embedAssets3"

# Create Milvus' schema
schema = milvus_client.create_schema(
    auto_id=False,
    enable_dynamic_field=True
)

# Adding new fields
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="chunk", datatype=DataType.VARCHAR, max_length=700)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=3072)

#Perform the indexes
index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name="id")

index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="IP"
)

#Create the collection
milvus_client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

#Embedding function
def get_embedding(text):
    model = "text-embedding-3-large"
    embedding = openai_client.embeddings.create(input=[text], model=model).data[0].embedding
    return embedding

#Test data
chunks = [
    "I_AI_IP_Misiones_IB_Alicuotas_2000 CLASIFICACIÓN DE LAS ACTIVIDADES ECONÓMICAS EFECTUADA MEDIANTE LA RG 6/00 (APLICABLE DESDE EL 28/2/2000) DADO QUE A LA FECHA DE ESTA ACTUALIZACIÓN LA DGR NO HA ADECUADO LAS ALÍCUOTAS PERTINENTES, ROGAMOS VER LAS MISMAS EN LA LEY IMPOSITIVA Alícuota general: 2,5%  501110 Venta de vehículos automotores, excepto motocicletas  501190 Venta de vehículos automotores, nuevos n.c.p. (incluye casas rodantes, trailers, camiones, remolques, ambulancias, ómnibus,",
    "ómnibus, microbuses y similares, cabezas tractoras, etc.).  501210 Venta de autos, camionetas y utilitarios, usados (incluye taxis, jeeps, 4x4 y vehículos similares).  501290 Venta de vehículos automotores, usados n.c.p. (incluye casas rodantes,trailers, camiones, remolques, ambulancias, ómnibus,microbuses y similares, cabezas tractoras, etc.).  Mantenimiento y reparación de vehículos automotores, excepto motocicletas  502100 Lavado automático y manual.  502210 Reparación de cámaras y cubiertas",
    "cámaras y cubiertas (incluye reparación de llantas).  502220 Reparación de amortiguadores, alineación de dirección y balanceo deruedas.  502300 Instalación y reparación de lunetas y ventanillas, alarmas, cerraduras, radios,sistemas de climatización automotor y grabado de cristales (incluye instalación y reparación de parabrisas, aletas, burletes, colisas, levanta vidrios, parlantes y autoestéreos, aire acondicionado, alarmas y sirenas, etc.).  502400 Tapizado y retapizado.  502500 Reparaciones",
    "502500 Reparaciones eléctricas, del tablero e instrumental. Reparación y recarga de -baterías.  502600 Reparación y pintura de carrocerías; colocación de guardabarros y protecciones exteriores.  502910 Instalación y reparaciónd e caños de escape.  502920 Mantenimiento y reparación de frenos.  502990 Mantenimiento y reparación del motor n.c.p.; mecánica integral (incluye auxilio y servicios de grúa para automotores;instalación y reparación de equipos de GNC).  Venta de partes, piezas y",
    "de partes, piezas y accesorios de vehículos automotores  503100 Venta al por mayor departes, piezas y accesorios de vehículos automotores.  503210 Venta al por menor de cámara y cubiertas.  503220 Venta al por menor de baterías.  503290 Venta al por menor departes, piezas y accesorios excepto cámaras, cubiertas y baterías.  Venta, mantenimiento y reparación de motocicletas y de sus partes, piezas y accesorios  504010 Venta de motocicletas y de sus partes, piezas y accesorios.  504020"
]

#Get the embedding from the context
datas = [{
    "id": i,
    "chunk": chunks[i],
    "vector": get_embedding(chunks[i])
} for i in range(0, len(chunks))]


print("Embeddings finalizados")
#Insert the embeddings to the milvus' collection
milvus_client.insert(collection_name=collection_name, data=datas)

print("Embeddings guardados")

# QUERY VECTOR SEARCH
question = "¿Se actualizo la alicuota en el DNU?"
vector = get_embedding(question)

#Perform the search
res = milvus_client.search(
    collection_name=collection_name,
    data=[vector],
    search_params={"metric_type": "IP", "params": {}},
    output_fields=['chunk']
)

data = json.dumps(res, indent=4)
print(data)