
import matplotlib.pyplot as plt

# Datos de la tabla
N = [1000, 2000, 4000, 8000, 16000, 32000, 44000]
t_seq = [0.0014, 0.0046, 0.0104, 0.0160, 0.0258, 0.0523, 0.0773]
t_inv = [0.0016, 0.0041, 0.0102, 0.0146, 0.0300, 0.0613, 0.1121]

plt.figure(figsize=(8, 5))

plt.plot(N, t_seq, marker="o", label="KNN secuencial")
plt.plot(N, t_inv, marker="s", label="KNN indexado")

plt.xlabel("Número de imágenes (N)")
plt.ylabel("Tiempo promedio por consulta (s)")
plt.title("Comparación de tiempos KNN secuencial vs KNN indexado")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("knn_times.png", dpi=300)

plt.show()