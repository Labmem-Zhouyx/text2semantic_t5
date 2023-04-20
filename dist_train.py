import os
import subprocess
import argparse


def main(args):
    processes = []
    for rank in range(args.num_gpus):
        my_env = os.environ.copy()
        if rank == 0:
            stdout = None
        else:
            stdout = open(os.devnull, 'w')
        command = [
                'python3',
                'train.py',
                '--config={}'.format(args.config),
                '--save_path={}'.format(args.save_path),
                '--meta_path={}'.format(args.meta_path),
                '--restore_path={}'.format(args.restore_path),
                '--rank={}'.format(rank),
                '--num_gpus={}'.format(args.num_gpus),
        ]
        p = subprocess.Popen(command, stdout=stdout, env=my_env)
        processes.append(p)
        print(command)

    for p in processes:
         p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Config yaml')
    
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help='Path to save checkpoints')
    parser.add_argument('--meta_path',
                        type=str,
                        required=True,
                        help='Path of metadata')
    parser.add_argument('--restore_path',
                        default=None,
                        type=str,
                        help='restore checkpoints')
    parser.add_argument('--num_gpus', default=0, type=int, help='rank')

    args = parser.parse_args()
    main(args)


