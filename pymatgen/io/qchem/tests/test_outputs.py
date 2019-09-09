# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import os
import math
import unittest

from monty.serialization import loadfn, dumpfn
from pymatgen.io.qchem.outputs import (QCOutput,
                                       QCStringfileParser,
                                       QCPerpGradFileParser,
                                       QCVFileParser)
from pymatgen.util.testing import PymatgenTest
try:
    import openbabel
    have_babel = True
except ImportError:
    have_babel = False

__author__ = "Samuel Blau, Brandon Wood, Shyam Dwaraknath"
__copyright__ = "Copyright 2018, The Materials Project"
__version__ = "0.1"

single_job_dict = loadfn(os.path.join(
    os.path.dirname(__file__), "single_job.json"))
multi_job_dict = loadfn(os.path.join(
    os.path.dirname(__file__), "multi_job.json"))
test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        'test_files', "molecules")

property_list = {"errors",
                 "multiple_outputs",
                 "completion",
                 "unrestricted",
                 "using_GEN_SCFMAN",
                 "final_energy",
                 "S2",
                 "optimization",
                 "energy_trajectory",
                 "opt_constraint",
                 "frequency_job",
                 "charge",
                 "multiplicity",
                 "species",
                 "initial_geometry",
                 "initial_molecule",
                 "SCF",
                 "Mulliken",
                 "optimized_geometry",
                 "optimized_zmat",
                 "molecule_from_optimized_geometry",
                 "last_geometry",
                 "molecule_from_last_geometry",
                 "geometries",
                 "gradients",
                 "frequency_mode_vectors",
                 "walltime",
                 "cputime",
                 "point_group",
                 "frequencies",
                 "IR_intens",
                 "IR_active",
                 "g_electrostatic",
                 "g_cavitation",
                 "g_dispersion",
                 "g_repulsion",
                 "total_contribution_pcm",
                 "ZPE",
                 "trans_enthalpy",
                 "vib_enthalpy",
                 "rot_enthalpy",
                 "gas_constant",
                 "trans_entropy",
                 "vib_entropy",
                 "rot_entropy",
                 "total_entropy",
                 "total_enthalpy",
                 "warnings",
                 "SCF_energy_in_the_final_basis_set",
                 "Total_energy_in_the_final_basis_set",
                 "solvent_method",
                 "solvent_data",
                 "using_dft_d3",
                 "single_point_job",
                 "force_job",
                 "freezing_string_job",
                 "pcm_gradients",
                 "CDS_gradients",
                 "RESP",
                 "trans_dip",
                 "string_num_images",
                 "string_energies",
                 "string_relative_energies",
                 "string_geometries",
                 "string_molecules",
                 "string_absolute_distances",
                 "string_proportional_distances",
                 "string_gradient_magnitudes",
                 "string_max_energy",
                 "string_ts_guess",
                 "string_initial_reactant_molecules",
                 "string_initial_product_molecules",
                 "string_initial_reactant_geometry",
                 "string_initial_product_geometry"}

if have_babel:
    property_list.add("structure_change")

single_job_out_names = {"unable_to_determine_lambda_in_geom_opt.qcout",
                        "thiophene_wfs_5_carboxyl.qcout",
                        "hf.qcout",
                        "hf_opt_failed.qcout",
                        "no_reading.qcout",
                        "exit_code_134.qcout",
                        "negative_eigen.qcout",
                        "insufficient_memory.qcout",
                        "freq_seg_too_small.qcout",
                        "crowd_gradient_number.qcout",
                        "quinoxaline_anion.qcout",
                        "tfsi_nbo.qcout",
                        "crowd_nbo_charges.qcout",
                        "h2o_aimd.qcout",
                        "quinoxaline_anion.qcout",
                        "crowd_gradient_number.qcout",
                        "bsse.qcout",
                        "thiophene_wfs_5_carboxyl.qcout",
                        "time_nan_values.qcout",
                        "pt_dft_180.0.qcout",
                        "qchem_energies/hf-rimp2.qcout",
                        "qchem_energies/hf_b3lyp.qcout",
                        "qchem_energies/hf_ccsd(t).qcout",
                        "qchem_energies/hf_cosmo.qcout",
                        "qchem_energies/hf_hf.qcout",
                        "qchem_energies/hf_lxygjos.qcout",
                        "qchem_energies/hf_mosmp2.qcout",
                        "qchem_energies/hf_mp2.qcout",
                        "qchem_energies/hf_qcisd(t).qcout",
                        "qchem_energies/hf_riccsd(t).qcout",
                        "qchem_energies/hf_tpssh.qcout",
                        "qchem_energies/hf_xyg3.qcout",
                        "qchem_energies/hf_xygjos.qcout",
                        "qchem_energies/hf_wb97xd_gen_scfman.qcout",
                        "new_qchem_files/pt_n2_n_wb_180.0.qcout",
                        "new_qchem_files/pt_n2_trip_wb_90.0.qcout",
                        "new_qchem_files/pt_n2_gs_rimp2_pvqz_90.0.qcout",
                        "new_qchem_files/VC_solv_eps10.2.qcout",
                        "crazy_scf_values.qcout",
                        "new_qchem_files/N2.qcout",
                        "new_qchem_files/julian.qcout",
                        "new_qchem_files/Frequency_no_equal.qout",
                        "new_qchem_files/gdm.qout",
                        "new_qchem_files/DinfH.qout",
                        "new_qchem_files/mpi_error.qout",
                        "new_qchem_files/molecule_read_error.qout",
                        "new_qchem_files/Optimization_no_equal.qout",
                        "new_qchem_files/2068.qout",
                        "new_qchem_files/2620.qout",
                        "new_qchem_files/1746.qout",
                        "new_qchem_files/1570.qout",
                        "new_qchem_files/1570_2.qout",
                        "new_qchem_files/single_point.qout",
                        "new_qchem_files/fsm.qout"}

multi_job_out_names = {"not_enough_total_memory.qcout",
                       "new_qchem_files/VC_solv_eps10.qcout",
                       "new_qchem_files/MECLi_solv_eps10.qcout",
                       "pcm_solvent_deprecated.qcout",
                       "qchem43_batch_job.qcout",
                       "ferrocenium_1pos.qcout",
                       "CdBr2.qcout",
                       "killed.qcout",
                       "aux_mpi_time_mol.qcout",
                       "new_qchem_files/VCLi_solv_eps10.qcout"}


class TestQCOutput(PymatgenTest):

    @staticmethod
    def generate_single_job_dict():
        """
        Used to generate test dictionary for single jobs.
        """
        single_job_dict = dict()
        for file in single_job_out_names:
            single_job_dict[file] = QCOutput(os.path.join(test_dir, file)).data
        dumpfn(single_job_dict, "single_job.json")

    @staticmethod
    def generate_multi_job_dict():
        """
        Used to generate test dictionary for multiple jobs.
        """
        multi_job_dict = dict()
        for file in multi_job_out_names:
            outputs = QCOutput.multiple_outputs_from_file(
                QCOutput, os.path.join(test_dir, file), keep_sub_files=False)
            data = []
            for sub_output in outputs:
                data.append(sub_output.data)
            multi_job_dict[file] = data
        dumpfn(multi_job_dict, "multi_job.json")

    def _test_property(self, key, single_outs, multi_outs):
        for name, outdata in single_outs.items():
            try:
                self.assertEqual(outdata.get(key), single_job_dict[name].get(key))
            except ValueError:
                self.assertArrayEqual(outdata.get(key), single_job_dict[name].get(key))
        for name, outputs in multi_outs.items():
            for ii, sub_output in enumerate(outputs):
                try:
                    self.assertEqual(sub_output.data.get(key), multi_job_dict[name][ii].get(key))
                except ValueError:
                    self.assertArrayEqual(sub_output.data.get(key), multi_job_dict[name][ii].get(key))

    def test_all(self):
        single_outs = dict()
        for file in single_job_out_names:
            single_outs[file] = QCOutput(os.path.join(test_dir, file)).data

        multi_outs = dict()
        for file in multi_job_out_names:
            multi_outs[file] = QCOutput.multiple_outputs_from_file(QCOutput,
                                                                   os.path.join(test_dir, file),
                                                                   keep_sub_files=False)

        for key in property_list:
            print('Testing ', key)
            self._test_property(key, single_outs, multi_outs)


class QCVfileParserTest(PymatgenTest):

    def test_init(self):
        filename = os.path.join(test_dir, "new_qchem_files", "Vfile.txt")

        with open(filename) as vfile:
            text = vfile.read()

        absolute_distances = [0.0, 1.07482, 2.21674, 3.29777, 4.41737, 5.54168,
                              6.59763, 7.70808, 8.80552, 9.86112, 10.88407,
                              11.94403, 13.00584, 14.02017, 14.26367, 16.13259,
                              17.17187, 18.21893, 19.26674, 20.29805, 21.37902,
                              22.43943, 23.55714, 24.63218, 25.77714, 26.84866,
                              27.93978, 29.03211, 30.11024]

        image_energies = [-1030.16113, -1030.16343, -1030.14377, -1030.13744,
                          -1030.13427, -1030.13045, -1030.10962, -1030.10316,
                          -1030.09018, -1030.06736, -1030.03444, -1029.98983,
                          -1029.9469, -1029.92152, -1029.92659, -1029.93749,
                          -1029.96343, -1030.00691, -1030.04057, -1030.06393,
                          -1030.07394, -1030.07758, -1030.08325, -1030.08465,
                          -1030.07883, -1030.06548, -1030.06812, -1030.14984,
                          -1030.22144]

        proportional_distances = [0.0, 0.0357, 0.07362, 0.10952, 0.14671,
                                  0.18405, 0.21912, 0.256, 0.29244, 0.3275,
                                  0.36147, 0.39668, 0.43194, 0.46563, 0.47371,
                                  0.53578, 0.5703, 0.60507, 0.63987, 0.67412,
                                  0.71002, 0.74524, 0.78236, 0.81807, 0.85609,
                                  0.89168, 0.92792, 0.96419, 1.0]

        relative_energies = [0.0, -1.44895, 10.89212, 14.86526, 16.84923,
                             19.24933, 32.32105, 36.37664, 44.51746, 58.83892,
                             79.4976, 107.49213, 134.42978, 150.35218,
                             147.17553, 140.33325, 124.05361, 96.76958,
                             75.64735, 60.99272, 54.7079, 52.42574, 48.87046,
                             47.98997, 51.64355, 60.01858, 58.3643, 7.08018,
                             -37.84533]

        parsed = QCVFileParser(filename=filename)

        self.assertEqual(parsed.filename, filename)
        self.assertEqual(parsed.text, text)
        self.assertEqual(parsed.data["num_images"], 29)
        self.assertSequenceEqual(parsed.data["absolute_distances"],
                                 absolute_distances)
        self.assertSequenceEqual(parsed.data["image_energies"], image_energies)
        self.assertSequenceEqual(parsed.data["proportional_distances"],
                                 proportional_distances)
        self.assertSequenceEqual(parsed.data["relative_energies"],
                                 relative_energies)


class QCStringfileParserTest(PymatgenTest):

    def test_init(self):
        filename = os.path.join(test_dir, "new_qchem_files", "stringfile.txt")

        # To generate JSON - uncomment if/when changes need to be made
        # data = QCStringfileParser(filename).data
        # dumpfn(data, "frozen_strings.json")

        # To load JSON
        frozen_string_dict = loadfn(os.path.join(
            os.path.dirname(__file__), "frozen_strings.json"))

        with open(filename) as stringfile:
            text = stringfile.read()

        parsed = QCStringfileParser(filename)

        self.assertEqual(parsed.filename, filename)
        self.assertEqual(parsed.text, text)

        keys = ["num_images", "length", "image_energies", "species",
                "geometries", "molecules"]
        for key in keys:
            try:
                self.assertEqual(parsed.data.get(key),
                                 frozen_string_dict.get(key))
            except ValueError:
                self.assertArrayEqual(parsed.data.get(key),
                                      frozen_string_dict.get(key))


class QCPerpGradFileParserTest(PymatgenTest):

    def test_init(self):
        filename = os.path.join(test_dir, "new_qchem_files",
                                "perp_grad_file.txt")

        with open(filename) as perp_grad_file:
            text = perp_grad_file.read()

        absolute_distances = [0.0, 1.07482, 2.21674, 3.29777, 4.41737, 5.54168,
                              6.59763, 7.70808, 8.80552, 9.86112, 10.88407,
                              11.94403, 13.00584, 14.02017, 14.26367, 16.13259,
                              17.17187, 18.21893, 19.26674, 20.29805, 21.37902,
                              22.43943, 23.55714, 24.63218, 25.77714, 26.84866,
                              27.93978, 29.03211, 30.11024]

        proportional_distances = [0.0, 0.0357, 0.07362, 0.10952, 0.14671,
                                  0.18405, 0.21912, 0.256, 0.29244, 0.3275,
                                  0.36147, 0.39668, 0.43194, 0.46563, 0.47371,
                                  0.53578, 0.5703, 0.60507, 0.63987, 0.67412,
                                  0.71002, 0.74524, 0.78236, 0.81807, 0.85609,
                                  0.89168, 0.92792, 0.96419, 1.0]

        gradient_magnitudes = [math.inf, 0.10157, 0.22465, 0.19255, 0.13654,
                               0.11796, 0.18782, 0.18502, 0.17487, 0.20797,
                               0.25543, 0.34664, 0.38776, 0.42603, 0.36126,
                               0.38034, 0.36734, 0.29388, 0.24044, 0.16646,
                               0.17634, 0.1228, 0.11449, 0.18048, 0.21151,
                               0.29764, 0.23548, 0.15584, math.inf]

        parsed = QCPerpGradFileParser(filename=filename)

        self.assertEqual(parsed.filename, filename)
        self.assertEqual(parsed.text, text)
        self.assertEqual(parsed.data["num_images"], 29)
        self.assertSequenceEqual(parsed.data["absolute_distances"],
                                 absolute_distances)
        self.assertSequenceEqual(parsed.data["proportional_distances"],
                                 proportional_distances)
        self.assertSequenceEqual(parsed.data["gradient_magnitudes"],
                                 gradient_magnitudes)


if __name__ == "__main__":
    unittest.main()
