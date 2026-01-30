#!/usr/bin/env python3

# written by sdvillal / StrawLab: https://github.com/strawlab/bag2hdf5/blob/master/bag2hdf5
# copied here for convenience

import os
import sys

import numpy as np
import argparse
import h5py

try:
    import rosbag
except:
    print('Cannot import rosbag... try installing bagpy (pip install bagpy)')

import warnings
import progressbar

FLOAT_TIME = True  # False to prevent saving of float redundant timestamps

def get_filenames(path, contains, does_not_contain=['~', '.pyc']):
    cmd = 'ls ' + '"' + path + '"'
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
    filelist = []
    for i, filename in enumerate(all_filelist):
        if contains in filename:
            fileok = True
            for nc in does_not_contain:
                if nc in filename:
                    fileok = False
            if fileok:
                filelist.append( os.path.join(path, filename) )
    return filelist

def flatten_msg(msg, t, max_strlen=None):
    assert max_strlen is not None  # don't accept default
    result = []
    for i, attr in enumerate(msg.__slots__):
        rostype = msg._slot_types[i]

        if attr == 'header':
            h = msg.header
            result.extend([h.seq,
                           h.stamp.secs,
                           h.stamp.nsecs,
                           h.frame_id,
                           ])
            if FLOAT_TIME:
                result.append(h.stamp.secs + h.stamp.nsecs * 1e-9)

        elif rostype == 'time':
            p = getattr(msg, attr)
            result.extend([p.secs, p.nsecs])
            if FLOAT_TIME:
                result.append(p.secs + p.nsecs * 1e-9)

        elif rostype == 'geometry_msgs/Point':
            p = getattr(msg, attr)
            result.extend([p.x, p.y, p.z])

        elif rostype == 'geometry_msgs/Quaternion':
            p = getattr(msg, attr)
            result.extend([p.x, p.y, p.z, p.w])

        elif rostype == 'geometry_msgs/Vector3':
            p = getattr(msg, attr)
            result.extend([p.x, p.y, p.z])

        elif rostype == 'geometry_msgs/Pose':
            p = getattr(msg, attr)
            result.extend([p.position.x, p.position.y, p.position.z,
                          p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])

        elif rostype == 'geometry_msgs/Twist':
            p = getattr(msg, attr)
            result.extend([p.linear.x, p.linear.y, p.linear.z,
                          p.angular.x, p.angular.y, p.angular.z])

        elif rostype == 'geometry_msgs/Accel':
            p = getattr(msg, attr)
            result.extend([p.linear.x, p.linear.y, p.linear.z,
                          p.angular.x, p.angular.y, p.angular.z])

        elif rostype == 'std_msgs/MultiArrayLayout':
            pass

        elif '[' in rostype and ']' in rostype and 'string' not in rostype:
            # Handle both dynamic ([]) and fixed-length ([N]) arrays
            p = getattr(msg, attr)
            l = [i for i in p]
            result.extend(l)

        else:
            p = getattr(msg, attr)
            if rostype == 'string' or rostype == 'string[]':
                if rostype == 'string[]':
                    # List of strings gets joined to one string
                    warnings.warn('string array is joined to single string', RuntimeWarning, stacklevel=2)
                    p = ','.join(p)
                assert len(p) <= max_strlen
            result.append(p)
    # also do timestamp
    result.extend([t.secs, t.nsecs])
    if FLOAT_TIME:
        result.append(t.secs + t.nsecs * 1e-9)

    return tuple(result)


def rostype2dtype(rostype, max_strlen=None):
    assert max_strlen is not None  # don't accept default

    # Handle fixed-length arrays like float64[9]
    if '[' in rostype and ']' in rostype and rostype != 'string[]' and rostype != 'bool[]':
        # Extract base type (e.g., 'float64' from 'float64[9]')
        basetype = rostype.split('[')[0]
        # Recursively get the dtype for the base type
        return rostype2dtype(basetype, max_strlen=max_strlen)
    
    if rostype == 'float32':
        dtype = np.float32
    elif rostype == 'float64':
        dtype = np.float64
    elif rostype == 'uint8' or rostype == 'byte':
        dtype = np.uint8
    elif rostype == 'uint16':
        dtype = np.uint16
    elif rostype == 'uint32':
        dtype = np.uint32
    elif rostype == 'uint64':
        dtype = np.uint64
    elif rostype == 'int8':
        dtype = np.int8
    elif rostype == 'int16':
        dtype = np.int16
    elif rostype == 'int32':
        dtype = np.int32
    elif rostype == 'int64':
        dtype = np.int64
    elif rostype == 'bool' or rostype == 'bool[]':
        dtype = np.bool
    elif rostype == 'string' or rostype == 'string[]':
        dtype = 'S' + str(max_strlen)
    else:
        raise ValueError('unknown ROS type: %s' % rostype)
    return dtype


def make_dtype(msg, max_strlen=None):
    assert max_strlen is not None  # don't accept default

    result = []
    for i, attr in enumerate(msg.__slots__):
        rostype = msg._slot_types[i]

        # Check for dynamic arrays first
        if '[]' in rostype and 'string' not in rostype:
            p = getattr(msg, attr)
            length_of_msg = len(p)
        # Check for fixed-length arrays like float64[9]
        elif '[' in rostype and ']' in rostype and 'string' not in rostype:
            p = getattr(msg, attr)
            length_of_msg = len(p)

        if rostype == 'Header' or rostype == 'std_msgs/Header':
            result.extend([('header_seq', np.uint32),
                           ('header_stamp_secs', np.int32),
                           ('header_stamp_nsecs', np.int32),
                           ('header_frame_id', 'S' + str(max_strlen))])
            if FLOAT_TIME:
                result.append(('header_stamp', np.float64))
        elif rostype == 'time':
            result.extend([('time_secs', np.int32),
                           ('time_nsecs', np.int32)])
            if FLOAT_TIME:
                result.append(('time', np.float64))
        elif rostype == 'geometry_msgs/Point':
            result.extend([(attr + '_x', np.float32),
                           (attr + '_y', np.float32),
                           (attr + '_z', np.float32),
                           ])
        elif rostype == 'geometry_msgs/Quaternion':
            result.extend([(attr + '_x', np.float32),
                           (attr + '_y', np.float32),
                           (attr + '_z', np.float32),
                           (attr + '_w', np.float32),
                           ])
        elif rostype == 'geometry_msgs/Vector3':
            result.extend([(attr + '_x', np.float64),
                           (attr + '_y', np.float64),
                           (attr + '_z', np.float64),
                           ])
        elif rostype == 'geometry_msgs/Pose':
            result.extend([(attr + '_position_x', np.float64),
                           (attr + '_position_y', np.float64),
                           (attr + '_position_z', np.float64),
                           (attr + '_orientation_x', np.float64),
                           (attr + '_orientation_y', np.float64),
                           (attr + '_orientation_z', np.float64),
                           (attr + '_orientation_w', np.float64),
                           ])
        elif rostype == 'geometry_msgs/Twist':
            result.extend([(attr + '_linear_x', np.float64),
                           (attr + '_linear_y', np.float64),
                           (attr + '_linear_z', np.float64),
                           (attr + '_angular_x', np.float64),
                           (attr + '_angular_y', np.float64),
                           (attr + '_angular_z', np.float64),
                           ])
        elif rostype == 'geometry_msgs/Accel':
            result.extend([(attr + '_linear_x', np.float64),
                           (attr + '_linear_y', np.float64),
                           (attr + '_linear_z', np.float64),
                           (attr + '_angular_x', np.float64),
                           (attr + '_angular_y', np.float64),
                           (attr + '_angular_z', np.float64),
                           ])
        elif rostype == 'std_msgs/MultiArrayLayout':
            pass
        # Handle both dynamic arrays ([]) and fixed-length arrays ([N])
        elif '[' in rostype and ']' in rostype and 'string' not in rostype:
            basetype = rostype.split('[')[0]
            r = []
            for j in range(length_of_msg):
                r.append((attr + '_' + str(j), np.__getattribute__(basetype)))
            result.extend(r)

        else:
            nptype = rostype2dtype(rostype, max_strlen=max_strlen)
            result.append((attr, nptype))
    # also do timestamp
    result.extend([('t_secs', np.int32), ('t_nsecs', np.int32)])
    if FLOAT_TIME:
        result.append(('t', np.float64))
    return result


def h5append(dset, arr):
    n_old_rows = dset.shape[0]
    n_new_rows = len(arr) + n_old_rows
    dset.resize(n_new_rows, axis=0)
    dset[n_old_rows:] = arr


def bag2hdf5(fname, out_fname, topics=None, max_strlen=None, skip_messages={}):
    assert max_strlen is not None  # don't accept default

    bag = rosbag.Bag(fname)
    results2 = {}
    chunksize = 10000
    dsets = {}

    # progressbar
    _pbw = ['converting %s: ' % fname, progressbar.Percentage()]
    pbar = progressbar.ProgressBar(widgets=_pbw, maxval=bag.size).start()

    if topics is None:
        print('AUTO FIND TOPICS')
        topics = []
        for topic, msg, t in bag.read_messages():
            topics.append(topic)
        topics = np.unique(topics).tolist()
        print(topics)

    print('skip messages: ')
    print(skip_messages)
    print

    try:
        with h5py.File(out_fname, mode='w') as out_f:
            for topic in topics:
                m = -1
                for topic, msg, t in bag.read_messages(topics=[topic]):
                    m += 1

                    if topic not in skip_messages.keys():
                        skip_messages[topic] = []

                    # print topic, m, skip_messages[topic]

                    # update progressbar
                    pbar.update(bag._file.tell())
                    # get the data

                    if m not in skip_messages[topic]:
                        this_row = flatten_msg(msg, t, max_strlen=max_strlen)

                        # convert it to numpy element (and dtype)
                        if topic not in results2:
                            try:
                                dtype = make_dtype(msg, max_strlen=max_strlen)
                            except:
                                print("*********************************")
                                print('topic:', topic)
                                print("\nerror while processing message:\n\n%r" % msg)
                                print('\nROW:', this_row)
                                print("*********************************")
                                raise
                            results2[topic] = dict(dtype=dtype,
                                                   object=[this_row])
                        else:
                            results2[topic]['object'].append(this_row)

                        # now flush our caches periodically
                        if len(results2[topic]['object']) >= chunksize:
                            arr = np.array(**results2[topic])
                            if topic not in dsets:
                                # initial creation
                                dset = out_f.create_dataset(topic, data=arr, maxshape=(None,),
                                                            compression='gzip',
                                                            compression_opts=9)
                                assert dset.compression == 'gzip'
                                assert dset.compression_opts == 9
                                dsets[topic] = dset
                            else:
                                # append to existing dataset
                                h5append(dsets[topic], arr)
                            del arr
                            # clear the cached values
                            results2[topic]['object'] = []

                    else:
                        print('skipping message: ', m)
            # done reading bag file. flush remaining data to h5 file
            for topic in results2:
                print(topic)
                print(results2[topic])
                print
                if not len(results2[topic]['object']):
                    # no data
                    continue
                arr = np.array(**results2[topic])
                if topic in dsets:
                    h5append(dsets[topic], arr)
                else:
                    out_f.create_dataset(topic,
                                         data=arr,
                                         compression='gzip',
                                         compression_opts=9)

    except:
        if os.path.exists(out_fname):
            os.unlink(out_fname)
        raise
    finally:
        pass
        pbar.finish()

    return out_fname

def preprocess_bag_file(data_path, name_contains='.bag'):
    topic = None
    max_strlen = 255

    bag_file = get_filenames(data_path, name_contains)[0]

    if not os.path.exists(bag_file):
        print('No file %s' % bag_file)
        sys.exit(1)
    fname = os.path.splitext(bag_file)[0]

    out_fname = fname + '.hdf5'
    if os.path.exists(out_fname):
        print('will not overwrite %s' % out_fname)
        return out_fname
        #sys.exit(1)
    else:
        out_fname = bag2hdf5(bag_file,
                             out_fname,
                             max_strlen=max_strlen,
                             topics=topic)
        return out_fname


def list_bag_topics(bag_filename):
    """
    List all topics in a ROS bag file without reading all messages.
    
    This is much faster than iterating through all messages because it only
    reads the bag file's metadata/index.
    
    Parameters:
    -----------
    bag_filename : str
        Path to the .bag file
    
    Returns:
    --------
    dict
        Dictionary with topic names as keys and info dictionaries as values.
        Each info dict contains: 'message_count', 'message_type', 'frequency'
    """
    if not os.path.exists(bag_filename):
        raise FileNotFoundError(f'No file {bag_filename}')
    
    bag = rosbag.Bag(bag_filename)
    
    # Get topic information from bag metadata
    topics_info = bag.get_type_and_topic_info()
    
    print(f"\nTopics in {bag_filename}:")
    print("=" * 80)
    print(f"{'Topic':<50} {'Type':<30} {'Count':<10}")
    print("-" * 80)
    
    result = {}
    for topic, info in topics_info.topics.items():
        print(f"{topic:<50} {info.msg_type:<30} {info.message_count:<10}")
        result[topic] = {
            'message_count': info.message_count,
            'message_type': info.msg_type,
            'frequency': info.frequency if hasattr(info, 'frequency') else None
        }
    
    print("=" * 80)
    print(f"Total topics: {len(result)}")
    print(f"Bag duration: {bag.get_end_time() - bag.get_start_time():.2f} seconds")
    print(f"Total messages: {sum(info['message_count'] for info in result.values())}")
    
    bag.close()
    return result


def main():
    """
    Main entry point for the bag2hdf5 command-line tool.
    This function is called when using the installed console script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help="the .bag file")
    parser.add_argument('--max_strlen', type=int, default=255,
                        help="maximum length of encoded strings")
    parser.add_argument('--out', type=str, default=None,
                        help="name of output file")
    parser.add_argument('--topic', type=str, nargs='*',
                        help="topic name to convert. defaults to all. "
                             "multiple may be specified.")
    parser.add_argument('--list-topics', action='store_true',
                        help="list topics in the bag file and exit")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print('No file %s' % args.filename)
        sys.exit(1)
    
    # If --list-topics flag is set, just list topics and exit
    if args.list_topics:
        list_bag_topics(args.filename)
        sys.exit(0)
    
    fname = os.path.splitext(args.filename)[0]
    if args.out is not None:
        output_fname = args.out
    else:
        output_fname = fname + '.hdf5'
        if os.path.exists(output_fname):
            print('will not overwrite %s' % output_fname)
            sys.exit(1)

    print('topic')
    print(args.topic)

    bag2hdf5(args.filename,
             output_fname,
             max_strlen=args.max_strlen,
             topics=args.topic)


if __name__ == '__main__':
    main()