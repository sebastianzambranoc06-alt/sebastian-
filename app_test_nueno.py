# app_test_nuevo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuración
st.set_page_config(
    page_title="Clasificador de Género IA",
    page_icon="👨‍👩‍👧‍👦",
    layout="wide"
)

# ================= FUNCIONES DE PREPROCESAMIENTO =================

def preparar_imagen(imagen):
    """Prepara la imagen para el modelo"""
    try:
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        # Redimensionar a 224x224 (tamaño esperado por la CNN)
        imagen = imagen.resize((224, 224))
        array_imagen = np.array(imagen)
        
        # Normalizar a [0, 1]
        array_imagen = array_imagen.astype('float32') / 255.0
        
        # Reordenar canales si es necesario y agregar dimensión batch
        array_imagen = np.transpose(array_imagen, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        lote_imagen = np.expand_dims(array_imagen, axis=0)
        
        return lote_imagen, np.transpose(array_imagen, (1, 2, 0))  # Para visualización
        
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None, None

# ================= FUNCIONES DE EXPLICABILIDAD SIMULADAS =================

def crear_heatmap_simple(centro_x=112, centro_y=112, tamano=50):
    """Crea un heatmap simple centrado en características faciales"""
    h, w = 224, 224
    y, x = np.ogrid[0:h, 0:w]
    
    # Región central principal (rostro)
    dist_centro = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)
    region_principal = np.exp(-dist_centro / (tamano * 1.5))
    
    # Regiones de ojos
    ojo_izq = np.exp(-((x - centro_x + 30)**2 + (y - centro_y - 20)**2) / (tamano * 8))
    ojo_der = np.exp(-((x - centro_x - 30)**2 + (y - centro_y - 20)**2) / (tamano * 8))
    
    # Región de boca
    boca = np.exp(-((x - centro_x)**2 + (y - centro_y + 30)**2) / (tamano * 12))
    
    # Combinar todo
    heatmap = (region_principal * 0.5 + 
               ojo_izq * 0.8 + ojo_der * 0.8 + 
               boca * 0.7)
    
    # Suavizar
    heatmap = suavizar_heatmap(heatmap, kernel_size=15)
    
    # Normalizar
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def suavizar_heatmap(matrix, kernel_size=5):
    """Suaviza un heatmap"""
    h, w = matrix.shape
    padded = np.pad(matrix, kernel_size//2, mode='edge')
    result = np.zeros_like(matrix)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.mean(patch)
    
    return result

def compute_saliency_map_simulado(imagen_procesada, prob_mujer):
    """Simula un saliency map basado en la probabilidad"""
    # Crear heatmap que se adapta a la predicción
    if prob_mujer > 0.5:
        # Para mujer: enfatizar características suaves
        centro_x, centro_y = 112, 100
        tamano = 45
    else:
        # Para hombre: características más angulares
        centro_x, centro_y = 112, 105  
        tamano = 40
    
    saliency = crear_heatmap_simple(centro_x, centro_y, tamano)
    
    # Ajustar intensidad según confianza
    confianza = max(prob_mujer, 1 - prob_mujer)
    saliency = saliency * (0.3 + confianza * 0.7)
    
    return saliency

def compute_grad_cam_simulado(imagen_procesada, prob_mujer):
    """Simula Grad-CAM basado en la predicción"""
    if prob_mujer > 0.5:
        # Para mujer: áreas más suaves y redondeadas
        centro_x, centro_y = 112, 100
        tamano = 55
    else:
        # Para hombre: áreas más definidas
        centro_x, centro_y = 112, 108
        tamano = 42
    
    grad_cam = crear_heatmap_simple(centro_x, centro_y, tamano)
    
    # Hacer el heatmap más específico
    confianza = max(prob_mujer, 1 - prob_mujer)
    grad_cam = grad_cam * (0.4 + confianza * 0.6)
    
    return grad_cam

# ================= MODELO SIMULADO (para demo) =================

@st.cache_resource
def cargar_modelo_simulado():
    """Simula la carga de un modelo CNN"""
    st.info("🔧 Usando sistema de demostración - Cargando modelo simulado")
    
    # En una implementación real, aquí cargarías un modelo ONNX
    # Por ahora simulamos predicciones basadas en características de la imagen
    
    class ModeloSimulado:
        def predict(self, imagen_batch):
            # Simular predicción basada en características de color
            imagen = imagen_batch[0]  # Primera imagen del batch
            
            # Extraer características simples para simular CNN
            prom_rojo = np.mean(imagen[0])  # Canal rojo
            prom_verde = np.mean(imagen[1])  # Canal verde
            prom_azul = np.mean(imagen[2])  # Canal azul
            
            # Simular lógica simple (esto sería reemplazado por el modelo real)
            # En general, tonos de piel tienden a tener ciertos rangos
            prob_mujer = 0.5 + (prom_rojo - prom_verde) * 2  # Simulación simple
            
            # Asegurar que esté en [0, 1]
            prob_mujer = max(0.1, min(0.9, prob_mujer))
            
            return np.array([[prob_mujer]])
    
    return ModeloSimulado()

# ================= INTERFAZ PRINCIPAL =================

def main():
    st.title("🧠 Clasificador de Género IA + Explicabilidad")
    st.markdown("**Análisis con Saliency Maps y Grad-CAM**")
    
    # Cargar modelo
    modelo = cargar_modelo_simulado()
    
    st.success("✅ Sistema listo para análisis con explicabilidad!")
    st.warning("⚠️ **Modo demostración**: Usando sistema simulado. Para producción, cargar modelo ONNX.")
    
    # Subir imagen
    archivo = st.file_uploader("Sube una imagen facial", type=['jpg', 'jpeg', 'png'])
    
    if archivo is not None:
        imagen = Image.open(archivo)
        
        # Mostrar imagen original
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(imagen, caption="Imagen original", use_column_width=True)
        
        # Procesar y predecir
        with st.spinner("🔍 Analizando imagen y generando explicaciones..."):
            lote_imagen, imagen_procesada = preparar_imagen(imagen)
            
            if lote_imagen is not None:
                try:
                    # Predicción simulada
                    prediccion = modelo.predict(lote_imagen)
                    prob_mujer = float(prediccion[0, 0])
                    prob_mujer = max(0.0, min(1.0, prob_mujer))
                    prob_hombre = 1.0 - prob_mujer
                    
                    # Determinar clase
                    if prob_mujer > 0.5:
                        resultado = "MUJER 👩"
                        confianza = prob_mujer
                        color = "#e84393"
                    else:
                        resultado = "HOMBRE 👨"
                        confianza = prob_hombre
                        color = "#3498db"
                    
                    # Generar mapas de explicabilidad simulados
                    saliency_map = compute_saliency_map_simulado(imagen_procesada, prob_mujer)
                    grad_cam_map = compute_grad_cam_simulado(imagen_procesada, prob_mujer)
                    
                except Exception as e:
                    st.error(f"Error en predicción: {e}")
                    return
        
        # Mostrar resultados principales
        with col2:
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
                <h2>{resultado}</h2>
                <h3>Confianza: {confianza:.1%}</h3>
                <p>Mujer: {prob_mujer:.3f} | Hombre: {prob_hombre:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de probabilidades
        st.markdown("---")
        st.subheader("📊 Distribución de Probabilidades")
        
        fig_prob, ax_prob = plt.subplots(figsize=(8, 3))
        bars = ax_prob.bar(['Hombre', 'Mujer'], [prob_hombre, prob_mujer], 
                          color=['#3498db', '#e84393'], alpha=0.8)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel('Probabilidad')
        ax_prob.bar_label(bars, fmt='%.3f', padding=3, fontweight='bold')
        ax_prob.spines['top'].set_visible(False)
        ax_prob.spines['right'].set_visible(False)
        st.pyplot(fig_prob)
        
        # ================= MAPAS DE EXPLICABILIDAD =================
        st.markdown("---")
        st.subheader("🔍 Explicabilidad del Modelo")
        
        # Configuración de visualización
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            transparency = st.slider("Transparencia del overlay", 0.1, 0.9, 0.5, 0.1)
        with col_config2:
            colormap = st.selectbox("Mapa de colores", ['viridis', 'hot', 'plasma', 'jet'])
        
        # Crear visualizaciones
        col_map1, col_map2 = st.columns(2)
        
        with col_map1:
            st.markdown("### 🎯 Saliency Map")
            st.markdown("**Muestra qué píxeles influyen más en la decisión**")
            
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = ax1.imshow(saliency_map, cmap=colormap)
            ax1.set_title('Mapa de Saliencia')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            ax2.imshow(imagen_procesada)
            ax2.imshow(saliency_map, cmap=colormap, alpha=transparency)
            ax2.set_title('Overlay en Imagen')
            ax2.axis('off')
            
            st.pyplot(fig1)
            
            st.info("""
            **Interpretación Saliency Map:**
            - Las áreas más brillantes indican píxeles que más influyeron en la decisión
            - Muestra sensibilidad a nivel de píxel individual
            - Rojo/amarillo = alta influencia, Azul = baja influencia
            """)
        
        with col_map2:
            st.markdown("### 🔥 Grad-CAM")
            st.markdown("**Muestra qué regiones semánticas fueron importantes**")
            
            fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
            
            im2 = ax3.imshow(grad_cam_map, cmap=colormap)
            ax3.set_title('Mapa Grad-CAM')
            ax3.axis('off')
            plt.colorbar(im2, ax=ax3, shrink=0.8)
            
            ax4.imshow(imagen_procesada)
            ax4.imshow(grad_cam_map, cmap=colormap, alpha=transparency)
            ax4.set_title('Overlay en Imagen')
            ax4.axis('off')
            
            st.pyplot(fig2)
            
            st.info("""
            **Interpretación Grad-CAM:**
            - Las áreas rojas/amarillas muestran regiones que el modelo consideró importantes
            - Resalta características semánticas como ojos, nariz, boca
            - Basado en activaciones de capas convolucionales
            """)
        
        # Información adicional
        with st.expander("📋 Información Técnica"):
            st.markdown("""
            **Características del Sistema:**
            - **Arquitectura**: CNN simulada con explicabilidad
            - **Compatibilidad**: 100% con Streamlit Cloud
            - **Explicabilidad**: Mapas de calor simulados basados en características faciales
            - **Precisión**: Sistema de demostración - para producción usar modelo ONNX entrenado
            
            **Próximos pasos para producción:**
            1. Entrenar CNN con PyTorch/TensorFlow
            2. Exportar modelo a formato ONNX
            3. Cargar modelo ONNX en esta aplicación
            """)

if __name__ == "__main__":
    main()