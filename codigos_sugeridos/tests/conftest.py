import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# caché de Numba aparte para las pruebas
os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp(prefix='numba_cache_'))
