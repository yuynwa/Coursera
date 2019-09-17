#!/use/bin/env python
# -*- coding: utf-8 -*-

import io
from PIL import Image
import requests

if __name__ == '__main__':

    def download_img(url, filename):

        r = requests.get(url, timeout=10)

        if requests.codes.ok != r.status_code:
            assert False, 'Status cide errir: {}.'.format(r.status_code)

        with Image.open(io.BytesIO(r.content)) as img:

            # img.save(filename)
            p = './data/' + filename
            img.save(p)



    with open('./untitled.sql') as f:

        for l in f:
            url = l.split(',')[3][2:-1]
            name = url.split('/'[-1])[-1]


            download_img(url=url, filename=name)






