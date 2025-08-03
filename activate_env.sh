#!/bin/bash

# Script para activar el entorno virtual y instalar dependencias

echo "🚀 Activando entorno virtual trading-alpha..."

# Activar venv
source venv/bin/activate

# Verificar que estamos en el venv correcto
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Entorno virtual activado: $(basename $VIRTUAL_ENV)"
else
    echo "❌ Error: No se pudo activar el entorno virtual"
    exit 1
fi

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Instalar en modo desarrollo
echo "🔧 Instalando trading-alpha en modo desarrollo..."
pip install -e .

echo "🎉 ¡Setup completo! Ya puedes usar trading-alpha"
echo ""
echo "Para activar manualmente en el futuro:"
echo "source venv/bin/activate"