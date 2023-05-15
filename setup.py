from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT="-e ."

#this funtion runing every lib in new line
def get_requirments(file_path:str)->List[str]:
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        [req.replace("\n","")for req in requirments]

    if HYPHEN_E_DOT in requirments:
        requirments.remove(HYPHEN_E_DOT)

    return requirments

setup(
    name="ml_delivery_tme",
    version="0.01",
    author="pratik",
    author_email="pratikterkar@gmail.com",
    install_requires=get_requirments("requirments.txt"),
    package=find_packages()


)