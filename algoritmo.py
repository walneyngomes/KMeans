import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Suponha que temos o seguinte dataframe
data = {
    'Nome': ['Alice', 'Bob', 'Carlos', 'Diana', 'Eva', 'Frank', 'Grace', 'Henry', 'Isabel', 'John'],
    'Idade': [25, 30, 22, 35, 28, 40, 23, 29, 33, 31],
    'Sexo': ['F', 'M', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M']
}
df = pd.DataFrame(data)

# Codificação do sexo em valores numéricos
label_encoder = LabelEncoder()
df['Sexo_Codificado'] = label_encoder.fit_transform(df['Sexo'])

# Selecionando apenas as colunas de idade e sexo codificado para clusterização
X = df[['Idade', 'Sexo_Codificado']]

# Normalização dos dados
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Aplicação do algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_normalized)

# Análise dos clusters
print(df)

# Visualização dos clusters
plt.scatter(df['Idade'], df['Sexo_Codificado'], c=df['Cluster'], cmap='viridis')
plt.title('Clusterização de Clientes')
plt.xlabel('Idade')
plt.ylabel('Sexo (0=F, 1=M)')
plt.show()
