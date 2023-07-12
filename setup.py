from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT="-e ."
def get_requirment(file_path:str)->List[str]:
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readline()
        requirments=[req.replace('\n','') for req in requirments]

        if HYPHEN_E_DOT in requirments:
            requirments.remove(HYPHEN_E_DOT)
    
setup(
    name="Ml project",
    version="0.1",
    author_name="Pratik",
    author_email='pratikterkar@gmail.com',
    require_install=['pandas','numpy'],
    pacakges=find_packages()

)