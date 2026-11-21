"""One entry for every check and measurement: python run.py ACTION [options].

The same command works on Linux, macOS and Windows. Bend builds native code
on Linux and macOS; on Windows the native steps run inside WSL in this
checkout (BEND_WSL_DISTRO selects a distribution, otherwise the default).
Host steps use the Python that runs this file. Generated files go under out/.
"""

import argparse
import os
from pathlib import Path
import shutil
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WINDOWS = sys.platform == 'win32'
CASES = ['bus_320', 'bus_640', 'zidane_320', 'zidane_640']
INTEGRATION_ENTRIES = ['convolution', 'scalar_convolution', 'array_api', 'array_api_zero_workspace', 'tensor_flow']


def run(command, failure):
    """Run one step from the repository root and stop at the first failure."""
    if subprocess.run([str(part) for part in command], cwd=ROOT).returncode:
        sys.exit(f'{failure}. See out/results/.')


def python(script, *arguments, failure):
    run([sys.executable, '-B', script, *arguments], failure)


def bend(source, output):
    """Emit C with the pinned official compiler."""
    run(['node', 'scripts/bend_launcher.mjs', source, '-o', output], f'Bend could not compile {source}')
    python('scripts/check_generated_c.py', output, failure=f'{output} shares convolution constructors')


def linux(*command, failure):
    """Run a native step: directly, or inside WSL on Windows."""
    if WINDOWS:
        distribution = ['-d', os.environ['BEND_WSL_DISTRO']] if os.environ.get('BEND_WSL_DISTRO') else []
        command = ['wsl.exe', *distribution, '--cd', ROOT, '--exec', *command]
    run(command, failure)


def linux_python(script, *arguments, failure):
    """Run a Python tool where the native programs run."""
    if WINDOWS:
        linux('sh', '-c', '. scripts/linux_environment.sh && exec "$py" -B "$@"', 'python', script, *arguments, failure=failure)
    else:
        python(script, *arguments, failure=failure)


def proofs(options):
    python('scripts/audit_proofs.py', '--check', failure='The official checker did not accept every proof')


def strict(options):
    python('scripts/check_official_proofs.py', '--require-complete', failure='The strict proof check did not pass')


def layout(options):
    python('scripts/check_layout.py', failure='The layout check did not pass')


def dependencies(options):
    # Upstream resolves imports from POSIX paths, so Windows runs it in WSL.
    if WINDOWS:
        distribution = ['-d', os.environ['BEND_WSL_DISTRO']] if os.environ.get('BEND_WSL_DISTRO') else []
        command = ['wsl.exe', *distribution, '--cd', ROOT, '--exec', 'sh', '-c',
                   '. scripts/linux_environment.sh && exec bun scripts/bend_dependencies.ts "$@"', 'bend_dependencies']
    else:
        local_bun = ROOT / '.tools/bun/node_modules/@oven/bun-linux-x64/bin/bun'
        command = [local_bun if platform.machine() == 'x86_64' and local_bun.is_file() else 'bun', 'scripts/bend_dependencies.ts']
    # A query prints its own error, such as an ambiguous name.
    sys.exit(subprocess.run([str(part) for part in [*command, *options.query]], cwd=ROOT).returncode)


def summary(options):
    python('scripts/summarize_validation.py', failure='The summary could not be written')


def integration(options):
    (ROOT / 'out/build/integration').mkdir(parents=True, exist_ok=True)
    python('benchmarks/checks/generate_cases.py', failure='Integration cases could not be generated')
    python('benchmarks/checks/tensor_flow_cases.py', failure='The tensor flow cases could not be generated')
    for entry in INTEGRATION_ENTRIES:
        bend(f'out/checks/{entry}.bend', f'out/build/integration/{entry}.c')
    linux('sh', 'benchmarks/checks/run.sh', failure='Native integration did not pass')


def api(options):
    linux_python('benchmarks/checks/public_api_cases.py', failure='The public API cases did not pass')
    linux_python('benchmarks/checks/tensor_file_cases.py', failure='The tensor file cases did not pass')


def gradients(options):
    linux_python('benchmarks/checks/gradient_cases.py', failure='The gradient cases did not pass')
    linux_python('benchmarks/checks/training_cases.py', failure='The training cases did not pass')


def reference(case):
    """Export the official reference and fused weights once per case."""
    if not (ROOT / f'out/results/{case}/reference.json').is_file():
        image, size = case.split('_')
        python('examples/yolov5/prepare_reference.py', '--image', image, '--size', size, failure='The official reference could not be prepared')


def model(options):
    case, threads = options.case, options.threads
    reference(case)
    (ROOT / 'out/models').mkdir(parents=True, exist_ok=True)
    bend('examples/yolov5/yolov5n.bend', 'out/models/yolov5n.c')
    linux('sh', 'examples/yolov5/build_run.sh', case, threads, failure='The native model did not build or run')
    python('examples/yolov5/compare_results.py', case, '--threads', threads, failure='The numerical comparison did not pass')
    report = 'comparison_native.json' if threads == 1 else f'comparison_native_{threads}t.json'
    print(f'PASS: out/results/{case}/{report}')


def vit(options):
    if not (ROOT / 'out/results/vit_b16/reference.json').is_file():
        python('examples/vit_b16/prepare_reference.py', failure='The official reference could not be prepared')
    (ROOT / 'out/models').mkdir(parents=True, exist_ok=True)
    bend('examples/vit_b16/vit_b16.bend', 'out/models/vit_b16.c')
    linux('sh', 'examples/vit_b16/build_run.sh', options.threads, failure='The native model did not build or run')
    python('examples/vit_b16/compare_results.py', '--threads', options.threads, failure='The numerical comparison did not pass')


def checkpoint(options):
    (ROOT / 'out/models').mkdir(parents=True, exist_ok=True)
    bend('examples/yolov5/load_checkpoint.bend', 'out/models/load_checkpoint.c')
    linux('sh', 'examples/yolov5/load_checkpoint.sh', failure='The checkpoint loader did not build or run')
    python('examples/yolov5/compare_checkpoint.py', failure='The checkpoint comparison did not pass')
    # storage formats and layouts the official checkpoint does not hold
    python('examples/yolov5/checkpoint_cases.py', 'out/results/checkpoint/cases.pt', failure='The checkpoint cases could not be written')
    linux('sh', 'examples/yolov5/load_checkpoint.sh', 'out/results/checkpoint/cases.pt', 'cases', failure='The checkpoint loader did not run on the cases')
    python('examples/yolov5/compare_checkpoint.py', '--weights', 'out/results/checkpoint/cases.pt', '--loaded', 'out/results/checkpoint/cases/bend',
           failure='The checkpoint cases comparison did not pass')
    print('PASS: out/results/checkpoint/official/comparison.json, out/results/checkpoint/cases/comparison.json')


def training(options):
    linux_python('examples/resnet20/prepare.py', failure='CIFAR-10 or the initial weights could not be prepared')
    (ROOT / 'out/models').mkdir(parents=True, exist_ok=True)
    bend('examples/resnet20/resnet20.bend', 'out/models/resnet20.c')
    results = 'out/results/resnet20'
    if options.epochs:
        trained, twin = f'{results}/bend_{options.threads}t', f'{results}/torch_{options.threads}t'
        linux('sh', 'examples/resnet20/build_run.sh', options.threads, options.epochs, 390, 5, 10000, trained, twin, failure='The training did not build or run')
        linux_python('examples/resnet20/compare_results.py', options.epochs, trained, '--torch', twin, failure='The results could not be compared')
        return
    # two short epochs on 1, 2, 4 and 8 threads, and the second one again from the checkpoint of the first
    check = f'{results}/check'
    shutil.rmtree(ROOT / check, ignore_errors=True)
    short = [3, 1, 200]
    for threads in [1, 2, 4, 8]:
        twin = [f'{check}/torch'] if threads == 4 else []
        linux('sh', 'examples/resnet20/build_run.sh', threads, 2, *short, f'{check}/bend_{threads}t', *twin, failure='The training did not build or run')
    for epochs in [1, 2]:
        linux('sh', 'examples/resnet20/build_run.sh', 4, epochs, *short, f'{check}/resumed', failure='The resumed training did not run')
    folders = [f'{check}/bend_{threads}t' for threads in [1, 2, 4, 8]] + [f'{check}/resumed']
    linux_python('examples/resnet20/compare_results.py', 2, *folders, '--torch', f'{check}/torch', '--tolerance', '1e-5', failure='The training check did not pass')


def benchmark(options):
    (ROOT / 'out/build/cpu_convolution').mkdir(parents=True, exist_ok=True)
    bend('benchmarks/convolution/run_convolution.bend', 'out/build/cpu_convolution/bend_convolution.c')
    linux('sh', 'benchmarks/convolution/run_cpu.sh', failure='The convolution comparison did not pass')


def backends(options):
    # The validation also compares every backend with the example's own native output.
    if not (ROOT / 'out/results/bus_640/native/pred.npy').is_file():
        model(argparse.Namespace(case='bus_640', threads=1))
    python('benchmarks/backends/generate_graph.py', failure='The graph could not be exported')
    bend('examples/yolov5/yolov5n.bend', 'out/models/yolov5n.c')
    if options.cuda:
        python('benchmarks/model/generate_yolo_graph.py', '--case', 'bus_640', '--target', 'cuda', failure='The CUDA graph could not be generated')
        run(['node', 'scripts/bend_launcher.mjs', 'out/models/bus_640_cuda.bend', '-o', 'out/backends/bend_cuda.c'], 'Bend could not compile the CUDA model')
        linux('sh', 'benchmarks/backends/run_cpu.sh', failure='The CPU comparison did not pass')
        linux('sh', 'benchmarks/backends/run_cuda.sh', failure='The CUDA comparison did not pass')
        python('benchmarks/backends/validate.py', failure='The numerical verification did not pass')
        python('benchmarks/backends/write_report.py', failure='The report could not be written')
    else:
        linux('sh', 'benchmarks/backends/run_cpu.sh', 'libraries', failure='The CPU comparison did not pass')
        selected = ['numpy_1t', 'numpy_4t', 'bend_cpu', 'bend_cpu_4t', 'torch_cpu_1t', 'torch_cpu_4t']
        python('benchmarks/backends/validate.py', '--backends', *selected, failure='The numerical verification did not pass')
        python('benchmarks/backends/write_report.py', '--cpu-only', failure='The report could not be written')
    print('Report: out/results/backend_comparison.md')


ACTIONS = {
    'proofs': (proofs, 'check every proof with the official checker and confirm the root theorem'),
    'strict': (strict, 'the same, requiring an unmodified pinned compiler checkout'),
    'layout': (layout, 'source layout, names, imports, links and frozen hashes'),
    'integration': (integration, 'native convolution cases on 1, 2, 4 and 8 threads'),
    'api': (api, 'programs against the public API, compared bit for bit with NumPy'),
    'gradients': (gradients, 'convolution gradients, losses, one training step, batch normalization and 40 learning steps, compared bit for bit with the reference on 1, 2, 4 and 8 workers and the learning steps with PyTorch'),
    'model': (model, 'YOLOv5n: generate, compile, run and compare with the reference'),
    'vit': (vit, 'ViT-B/16: compile, run and compare with PyTorch'),
    'checkpoint': (checkpoint, 'load the official .pt checkpoint in Bend and compare every tensor with torch.load'),
    'benchmark': (benchmark, 'time single convolution shapes'),
    'backends': (backends, 'time the YOLOv5n forward on NumPy, Stelliferous and PyTorch'),
    'training': (training, 'ResNet-20 on CIFAR-10: two short epochs compared bit for bit on 1, 2, 4 and 8 threads and after a resume, and the first loss with PyTorch; --epochs E trains E full epochs in both'),
    'dependencies': (dependencies, 'where each Bend declaration and file is used: unused imports, unreached declarations, why NAME, users NAME, deps NAME'),
    'summary': (summary, 'list which recorded results passed, without rerunning them'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='action', required=True)
    for name, (function, text) in ACTIONS.items():
        command = commands.add_parser(name, help=text)
        command.set_defaults(function=function)
        if name == 'model':
            command.add_argument('--case', choices=CASES, default='bus_640')
            command.add_argument('--threads', type=int, default=1)
        if name == 'training':
            command.add_argument('--threads', type=int, default=4)
            command.add_argument('--epochs', type=int, default=0, help='full epochs to reach, continuing after the last one written')
        if name == 'vit':
            command.add_argument('--threads', type=int, default=1)
        if name == 'dependencies':
            command.add_argument('query', nargs=argparse.REMAINDER, help='unused [PREFIX], why NAME, users NAME or deps NAME; help lists them')
        if name == 'backends':
            command.add_argument('--cuda', action='store_true', help='also run the handwritten C and CUDA comparisons')
    options = parser.parse_args()
    if getattr(options, 'threads', 1) < 1:
        parser.error('--threads must be positive')
    os.environ['PYTHONPYCACHEPREFIX'] = str(ROOT / 'out/python_cache')
    if not WINDOWS:
        os.environ.setdefault('BENCH_PYTHON', sys.executable)
    options.function(options)


if __name__ == '__main__':
    main()
