import zipfile


zip = zipfile.ZipFile('./data/happy-or-sad.zip')
zip.extractall('./data/happy-or-sad')
zip.close()


