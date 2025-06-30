
import subprocess


def run_pflotran_main():
    bash_code = """
        #!/bin/bash
        shopt -s extglob
        base_dir="$(pwd)"
        mkdir -p "${base_dir}/src/RunPFLOTRAN/output"

        for i in {228..230}; do
            infile="${base_dir}/src/RunPFLOTRAN/input/sample_${i}.in"
            mpirun -n 36 $PFLOTRAN_DIR/src/pflotran/pflotran -input_prefix "${infile%.*}"
            output_subdir="${base_dir}/src/RunPFLOTRAN/output/$(basename ${infile%.*})"
            mkdir -p "${output_subdir}"
            mv ${base_dir}/src/RunPFLOTRAN/input/!(mesh).h5 "${output_subdir}"
            mv ${base_dir}/src/RunPFLOTRAN/input/*.xmf "${output_subdir}"
            mv ${base_dir}/src/RunPFLOTRAN/input/*.pft "${output_subdir}"
            mv ${base_dir}/src/RunPFLOTRAN/input/*.out "${output_subdir}"
            mv ${base_dir}/src/RunPFLOTRAN/input/sample*.dat "${output_subdir}"
            cp "${base_dir}/src/RunPFLOTRAN/input/sample_${i}.in" "${output_subdir}"
        done
        """
    subprocess.run(['bash', '-c', bash_code], check=True)


if __name__ == '__main__':

    run_pflotran_main()
