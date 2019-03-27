""" Source -- https://github.com/nperraud/download-celebA-HQ/ """

import requests
import tarfile
import zipfile
import gzip
import os
import hashlib
import sys
from glob import glob
from urllib.request import urlretrieve
from subprocess import Popen
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download celebA helper')
parser.add_argument('path', type=str)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, chunk_size=32 * 1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(
                response.iter_content(chunk_size),
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=destination):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def checksum(filename, method='sha1'):
    data = open(filename, 'rb').read()
    if method == 'sha1':
        return hashlib.sha1(data).hexdigest()
    elif method == 'md5':
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError('Invalid method: %s' % method)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def archive_extract(filepath, target_dir):
    target_dir = os.path.abspath(target_dir)
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r') as tarf:
            # Check that no files get extracted outside target_dir
            for name in tarf.getnames():
                abs_path = os.path.abspath(os.path.join(target_dir, name))
                if not abs_path.startswith(target_dir):
                    raise RuntimeError('Archive tries to extract files '
                                       'outside target_dir.')
            tarf.extractall(target_dir)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zipf:
            zipf.extractall(target_dir)
    elif filepath[-3:].lower() == '.gz':
        with gzip.open(filepath, 'rb') as gzipf:
            with open(filepath[:-3], 'wb') as outf:
                outf.write(gzipf.read())
    elif '.7z' in filepath:
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .7z '
                                      'compression.')
        print('Using 7z!!!')
        cmd = '7z x {} -o{}'.format(filepath, target_dir)
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError(
                'Archive file extraction failed for {}.'.format(filepath))
    elif filepath[-2:].lower() == '.z':
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .Z '
                                      'compression.')
        cmd = 'gzip -d {}'.format(filepath)
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError(
                'Archive file extraction failed for {}.'.format(filepath))
    else:
        raise ValueError('{} is not a supported archive file.'.format(filepath))


def download_and_check(drive_data, path):
    save_paths = list()
    n_files = len(drive_data["filenames"])
    for i in range(n_files):
        drive_id = drive_data["drive_ids"][i]
        filename = drive_data["filenames"][i]
        save_path = os.path.join(path, filename)
        require_dir(os.path.dirname(save_path))
        print('Downloading {} to {}'.format(filename, save_path))
        download_file_from_google_drive(drive_id, save_path)
        print('Done!')
        if "sha1" in drive_data:
            sha1 = drive_data["sha1"][i]
            print('Check SHA1 {}'.format(save_path))
            if sha1 != checksum(save_path, 'sha1'):
                raise RuntimeError('Checksum mismatch for %s.' % save_path)
        save_paths.append(save_path)
    return save_paths


def download_celabA(dataset_dir):
    _IMGS_DRIVE = dict(
            filenames = [
                'img_celeba.7z.001', 'img_celeba.7z.002', 'img_celeba.7z.003',
                'img_celeba.7z.004', 'img_celeba.7z.005', 'img_celeba.7z.006',
                'img_celeba.7z.007', 'img_celeba.7z.008', 'img_celeba.7z.009',
                'img_celeba.7z.010', 'img_celeba.7z.011', 'img_celeba.7z.012',
                'img_celeba.7z.013', 'img_celeba.7z.014'
                ],
            drive_ids = [
                '0B7EVK8r0v71pQy1YUGtHeUM2dUE', '0B7EVK8r0v71peFphOHpxODd5SjQ',
                '0B7EVK8r0v71pMk5FeXRlOXcxVVU', '0B7EVK8r0v71peXc4WldxZGFUbk0',
                '0B7EVK8r0v71pMktaV1hjZUJhLWM', '0B7EVK8r0v71pbWFfbGRDOVZxOUU',
                '0B7EVK8r0v71pQlZrOENSOUhkQ3c', '0B7EVK8r0v71pLVltX2F6dzVwT0E',
                '0B7EVK8r0v71pVlg5SmtLa1ZiU0k', '0B7EVK8r0v71pa09rcFF4THRmSFU',
                '0B7EVK8r0v71pNU9BZVBEMF9KN28', '0B7EVK8r0v71pTVd3R2NpQ0FHaGM',
                '0B7EVK8r0v71paXBad2lfSzlzSlk', '0B7EVK8r0v71pcTFwT1VFZzkzZk0'
                ]
            )

    _ALIGNED_IMGS_DRIVE = dict(
            filenames = [
                'img_align_celeba.zip'
                ],
            drive_ids = [
                '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
                ],
            sha1 = [
                'b7e1990e1f046969bd4e49c6d804b93cd9be1646'
                ]
            )

    _PARTITIONS_DRIVE = dict(
            filenames = [
                'Eval/list_eval_partition.txt'
                ],
            drive_ids = [
                '0B7EVK8r0v71pY0NSMzRuSXJEVkk'
                ],
            sha1 = [
                'fb3d89825c49a2d389601eacb10d73815fd3c52d'
                ]
            )

    _ALIGNED_ATTRIBUTES_DRIVE = dict(
            filenames = [
                'Anno/list_attr_celeba.txt'
                ],
            drive_ids = [
                '0B7EVK8r0v71pblRyaVFSWGxPY0U'
                ],
            sha1 = [
                '225788ff6c9d0b96dc21144147456e0388195617'
                ]
            )

    _ATTRIBUTES_DRIVE = dict(
            filenames = [
                'Anno/list_landmarks_celeba.txt'
                ],
            drive_ids = [
                '0B7EVK8r0v71pTzJIdlJWdHczRlU'
                ],
            sha1 = [
                'ea255cd0ffe98ca88bff23767f7a5ece7710db57'
                ]
            )

    n_imgs = 202599
    img_dir_align = os.path.join(dataset_dir, 'Img', 'img_align_celeba')
    img_dir = os.path.join(dataset_dir, 'Img', 'img_celeba')

    filepaths = download_and_check(_ALIGNED_IMGS_DRIVE, dataset_dir)

    filepath = filepaths[0]
    print('Extract archive {}'.format(filepath))
    archive_extract(filepath, os.path.join(dataset_dir, 'Img'))
    print('Done!')
    os.remove(filepath)

    n_imgsd = sum([1 for file in os.listdir(img_dir_align) if file[-4:] == '.jpg'])
    assert (n_imgsd == n_imgs)

    filepaths = download_and_check(_PARTITIONS_DRIVE, dataset_dir)

    filepaths = download_and_check(_ATTRIBUTES_DRIVE, dataset_dir)

    filepaths = download_and_check(_ALIGNED_ATTRIBUTES_DRIVE, dataset_dir)

    filepaths = download_and_check(_IMGS_DRIVE, dataset_dir)
    filepath = filepaths[0]
    print('Extract archive {}'.format(filepath))
    archive_extract(filepath, os.path.join(dataset_dir, 'Img'))
    print('Done!')

    for filepath in filepaths:
        print('Remove: {}'.format(filepath))
        os.remove(filepath)

    n_imgsd = len(glob(os.path.join(img_dir, '*.jpg')))
    assert (n_imgsd == n_imgs)

    return True

if __name__ == '__main__':
    args = parser.parse_args()
    dirpath = args.path
    dataset_dir = os.path.join(dirpath, 'CelebA')
    download_celabA(dataset_dir)




