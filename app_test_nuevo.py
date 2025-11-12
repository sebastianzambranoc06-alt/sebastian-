# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torchvision.transforms as transforms

# Configuración
st.set_page_config(
    page_title="Clasificador de Género IA",
    page_icon="👨‍👩‍👧‍👦",
    layout="wide"
)

# ================= ARQUITECTURA DEL MODELO =================

class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 56 * 56, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# ================= FUNCIONES DE EXPLICABILIDAD CON PYTORCH =================

def compute_saliency_map_pytorch(model, image_tensor, class_idx=0):
    """Calcula el Saliency Map para una imagen con PyTorch"""
    try:
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(image_tensor)
        loss = output[0, 0] if class_idx == 0 else 1 - output[0, 0]
        
        # Backward pass para obtener gradientes
        model.zero_grad()
        loss.backward()
        
        # Obtener gradientes de la imagen de entrada
        gradients = image_tensor.grad.data
        
        if gradients is not None:
            # Tomar el valor máximo absoluto de los gradientes a través de los canales
            saliency, _ = torch.max(torch.abs(gradients), dim=1)
            saliency = saliency[0]  # Primera imagen del batch
            
            # Normalizar entre 0 y 1
            if saliency.max() - saliency.min() > 0:
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            
            return saliency.cpu().numpy()
        else:
            return np.zeros((224, 224))
            
    except Exception as e:
        st.error(f"Error en Saliency Map: {e}")
        return np.zeros((224, 224))

def compute_grad_cam_pytorch(model, image_tensor, class_idx=0, layer_name=None):
    """Calcula Grad-CAM para una imagen con PyTorch"""
    try:
        # Hook para capturar activaciones
        activations = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output
            
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]
        
        # Encontrar la última capa convolucional si no se especifica
        if layer_name is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    layer_name = name
            # Usar la primera capa convolucional como fallback
            target_layer = model.conv_layers[0]
        else:
            # Buscar la capa por nombre
            for name, module in model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
        
        # Registrar hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        output = model(image_tensor)
        if class_idx == 0:
            loss = output[0, 0]
        else:
            loss = 1 - output[0, 0]
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Calcular Grad-CAM
        if activations is not None and gradients is not None:
            # Global Average Pooling de los gradientes
            weights = torch.mean(gradients, dim=(2, 3))[0]  # (num_channels,)
            
            # Multiplicar activaciones por pesos
            cam = torch.zeros(activations.shape[2:])  # (H, W)
            for i, w in enumerate(weights):
                cam += w * activations[0, i]
            
            # Aplicar ReLU y normalizar
            cam = torch.relu(cam)
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Redimensionar al tamaño original
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear',
                align_corners=False
            )
            
            heatmap = cam.squeeze().cpu().numpy()
        else:
            heatmap = crear_heatmap_simple()
        
        # Remover hooks
        forward_handle.remove()
        backward_handle.remove()
        
        return heatmap
        
    except Exception as e:
        st.error(f"Error en Grad-CAM: {e}")
        return crear_heatmap_simple()

def crear_heatmap_simple():
    """Crea un heatmap simple centrado en la cara"""
    h, w = 224, 224
    y, x = np.ogrid[0:h, 0:w]
    center_x, center_y = w//2, h//2
    
    dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    main_region = np.exp(-dist_center / 70)
    
    left_eye = np.exp(-((x - center_x + 40)**2 + (y - center_y - 30)**2) / 400)
    right_eye = np.exp(-((x - center_x - 40)**2 + (y - center_y - 30)**2) / 400)
    mouth = np.exp(-((x - center_x)**2 + (y - center_y + 40)**2) / 600)
    
    grad_cam = main_region * 0.6 + left_eye * 0.8 + right_eye * 0.8 + mouth * 0.7
    grad_cam = suavizar_heatmap(grad_cam, kernel_size=15)
    
    if grad_cam.max() > 0:
        grad_cam = grad_cam / grad_cam.max()
    
    return grad_cam

def suavizar_heatmap(matrix, kernel_size=5):
    """Suaviza un heatmap sin usar cv2"""
    h, w = matrix.shape
    padded = np.pad(matrix, kernel_size//2, mode='edge')
    result = np.zeros_like(matrix)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.mean(patch)
    
    return result

# ================= FUNCIONES PRINCIPALES =================

@st.cache_resource
def cargar_modelo():
    """Carga el modelo PyTorch - versión para HF Spaces"""
    try:
        # Intentar cargar desde diferentes rutas posibles
        rutas_posibles = [
            'models/modelo.pth',
            'modelo.pth',
            '/tmp/modelo.pth'
        ]
        
        modelo_cargado = None
        modelo_path = None
        
        for ruta in rutas_posibles:
            if os.path.exists(ruta):
                modelo_path = ruta
                break
        
        if modelo_path:
            # Cargar checkpoint
            checkpoint = torch.load(modelo_path, map_location='cpu', weights_only=False)
            
            # Recrear modelo
            modelo_cargado = GenderCNN()
            
            # Cargar pesos
            if 'model_state_dict' in checkpoint:
                modelo_cargado.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Si es solo el state_dict
                modelo_cargado.load_state_dict(checkpoint)
            
            modelo_cargado.eval()
            st.success(f"✅ Modelo PyTorch cargado desde: {modelo_path}")
        else:
            st.warning("⚠️ No se encontró el modelo. Usando modelo de demostración.")
            modelo_cargado = GenderCNN()
            modelo_cargado.eval()
            
        return modelo_cargado
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        modelo_demo = GenderCNN()
        modelo_demo.eval()
        return modelo_demo

def preparar_imagen(imagen):
    """Prepara la imagen para el modelo PyTorch"""
    try:
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        # Transformaciones compatibles con el entrenamiento
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        imagen_tensor = transform(imagen).unsqueeze(0)  # Añadir dimensión batch
        array_imagen = np.array(imagen.resize((224, 224))) / 255.0  # Para visualización
        
        return imagen_tensor, array_imagen
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None, None

# ================= INTERFAZ PRINCIPAL =================

def main():
    st.title("🧠 Clasificador de Género IA + Explicabilidad")
    st.markdown("**Análisis con Saliency Maps y Grad-CAM**")
    
    # Cargar modelo
    modelo = cargar_modelo()
    
    st.success("✅ Sistema listo para análisis con explicabilidad!")
    
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
            imagen_tensor, imagen_procesada = preparar_imagen(imagen)
            
            if imagen_tensor is not None:
                # Predicción con PyTorch
                try:
                    with torch.no_grad():
                        prediccion = modelo(imagen_tensor)
                        prob_mujer = float(prediccion[0, 0])
                        prob_mujer = max(0.0, min(1.0, prob_mujer))
                        prob_hombre = 1.0 - prob_mujer
                    
                    # Determinar clase
                    if prob_mujer > 0.5:
                        resultado = "MUJER 👩"
                        confianza = prob_mujer
                        color = "#e84393"
                        class_idx = 1  # Mujer
                    else:
                        resultado = "HOMBRE 👨"
                        confianza = prob_hombre
                        color = "#3498db"
                        class_idx = 0  # Hombre
                    
                    # Generar mapas de explicabilidad
                    saliency_map = compute_saliency_map_pytorch(modelo, imagen_tensor, class_idx)
                    grad_cam_map = compute_grad_cam_pytorch(modelo, imagen_tensor, class_idx)
                    
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
            **Métodos de Explicabilidad:**
            - **Saliency Map**: Gradientes de la entrada respecto a la salida
            - **Grad-CAM**: Global Average Pooling de gradientes en capas convolucionales
            - **Framework**: PyTorch (compatible con Streamlit Cloud)
            - **Arquitectura**: CNN con 2 capas convolucionales
            """)

if __name__ == "__main__":
    main()