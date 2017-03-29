import sys
import os
import urllib.request
import tarfile
import zipfile

def _print_download_progress(count, block_size, total_size):
    # Porcentaje de completacion.
    pct_complete = float(count * block_size) / total_size
    # -r opcion para imprimer sobre la misma linea.
    msg = "\r- Progreso de descarga: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract(url, download_dir):    
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)    
    if not os.path.exists(file_path):
        # Revisar si el directorio existe si no se crea.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        print("Comienza la descarga: " + filename)
        # Descarga desde internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)
        print() 
        print("Descargar terminarda.")
        if file_path.endswith(".zip"):
            # Desempaquetar zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Desempaquetar the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        print("Desempaquetamiento finalizado.")
    else:
        print("Data aparentemente descargada y desempaquetada")


if __name__ == "__main__":
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = "D:\\CesarBragagnini\\MCS\\SistemasInteligentes\\CNN\\data\\"
    maybe_download_and_extract(url, path)
