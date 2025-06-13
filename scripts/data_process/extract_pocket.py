"""
using prody to extract the pocket
"""
import os, re
import sys
import json
import shutil
import argparse
import prody as pr

from openbabel import openbabel as ob

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process CPSet')
    parser.add_argument('--src_root', type=str, required=True,
                        help='Input directory containing metadata and pdb_files')
    parser.add_argument('--tgt_root', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=10.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def write_file(output_file, outline):
    buffer = open(output_file, 'w')
    buffer.write(outline)
    buffer.close()


def lig_rename(infile, outfile):
    ##some peptides may impede the generation of pocket, so rename the ligname first.
    lines = open(infile, 'r').readlines()
    newlines = []
    for line in lines:
        if re.search(r'^HETATM|^ATOM', line):
            newlines.append(line[:17] + "LIG" + line[20:])
        else:
            newlines.append(line)
    write_file(outfile, ''.join(newlines))


def check_mol(infile, outfile):
    # Some metals may have the same ID as ligand, thus making ligand included in the pocket.
    os.system("cat %s | sed '/LIG/d' > %s"%(infile, outfile))


def extract_pocket(pdb_id, protpath, ligpath, cutoff=10.0, workdir='.'):
    """
    protpath: the path of protein file (.pdb).
    ligpath: the path of ligand file (.sdf|.mol2|.pdb).
    cutoff: the distance range within the ligand to determine the pocket.
    protname: the name of the protein.
    ligname: the name of the ligand.
    workdir: working directory.
    """
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats(ligpath.split('.')[-1], "pdb")
    
    ligname = f'{pdb_id}_ligand'
    protname = f'{pdb_id}_protein'

    if not re.search(r'.pdb$', ligpath):
        # convert ligand to pdb
        ligand = ob.OBMol()
        obConversion.ReadFile(ligand, ligpath)
        obConversion.WriteFile(ligand, "%s/%s.pdb"%(workdir, f'{pdb_id}_ligand'))
    
    xprot = pr.parsePDB(protpath)
    
    lig_rename("%s/%s.pdb"%(workdir, ligname), "%s/%s2.pdb"%(workdir, ligname))
    os.remove("%s/%s.pdb"%(workdir, ligname))
    os.rename("%s/%s2.pdb"%(workdir, ligname), "%s/%s.pdb"%(workdir, ligname)) 
    xlig = pr.parsePDB("%s/%s.pdb"%(workdir, ligname))
    lresname = xlig.getResnames()[0]
    xcom = xlig + xprot
    
    # select ONLY atoms that belong to the protein
    ret = xcom.select(f'same residue as exwithin %s of resname %s'%(cutoff, lresname))
    
    pr.writePDB("%s/%s_pocket_%s_temp.pdb"%(workdir, protname, cutoff), ret)
    
    check_mol("%s/%s_pocket_%s_temp.pdb"%(workdir, protname, cutoff), "%s/%s_pocket_%s.pdb"%(workdir, protname, cutoff))
    os.remove("%s/%s.pdb"%(workdir, ligname))
    os.remove("%s/%s_pocket_%s_temp.pdb"%(workdir, protname, cutoff))
    
    # copy ligand file
    dst_ligand_file = os.path.join(workdir, f'{pdb_id}_ligand.sdf')
    shutil.copy(ligpath, dst_ligand_file)


def main(args):
    src_root = args.src_root
    tgt_root = args.tgt_root
    
    # prepare target_ids
    label_ls = sorted([pdb_id for pdb_id in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, pdb_id))])
    print_log(f'Extracting ({len(label_ls)})...')    
    
    cnt, valid = 0, 0
    for pdb_id in label_ls:
        cnt += 1
        prot_path = os.path.join(src_root, pdb_id, f'{pdb_id}_protein.pdb')
        lig_path = os.path.join(src_root, pdb_id, f'{pdb_id}_ligand.sdf')
        try:
            workdir = os.path.join(tgt_root, pdb_id)
            os.makedirs(workdir, exist_ok=True)
            extract_pocket(pdb_id, prot_path, lig_path, cutoff=args.interface_dist_th, workdir=workdir)
            valid += 1
            print_log(f'{pdb_id} succeeded, valid/processed={valid}/{cnt}')
        except Exception as e:
            print_log(f'{pdb_id} extracting failed: {e}')
    
    print_log('Finished.')
    

if __name__ == '__main__':
    main(parse())