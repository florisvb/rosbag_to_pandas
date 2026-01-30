import pandas as pd
import h5py

def find_all_datasets(filename):
    """
    Recursively find all datasets in an HDF5 file.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    
    Returns:
    --------
    list of str
        List of all dataset paths in the file
    """
    def _find_datasets_recursive(group, prefix=''):
        """Helper function to recursively find datasets"""
        datasets = []
        for key in group.keys():
            item = group[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                datasets.append(path)
            elif isinstance(item, h5py.Group):
                datasets.extend(_find_datasets_recursive(item, path))
        return datasets
    
    with h5py.File(filename, 'r') as f:
        return _find_datasets_recursive(f)


def explore_hdf5_structure(filename, max_depth=3):
    '''
    Print the structure of an HDF5 file to help understand its organization.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    max_depth : int
        Maximum depth to explore
    '''
    def print_structure(name, obj, depth=0):
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}Dataset: {name} - shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}")
            if depth < max_depth:
                for key in obj.keys():
                    print_structure(key, obj[key], depth + 1)
    
    with h5py.File(filename, 'r') as f:
        print(f"HDF5 File: {filename}")
        print("="*60)
        for key in f.keys():
            print_structure(key, f[key], depth=0)
        print("="*60)


def get_pandas_dataframe_from_uncooperative_hdf5(filename, key='first_key'):
    '''
    Load data from HDF5 file created by bag2hdf5 into a pandas DataFrame.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    key : str or list of str
        The dataset path(s) to load. Options:
        - 'first_key': Load the first dataset found (recursively)
        - 'all_keys': Load all datasets and return a dictionary of DataFrames
        - Full path to dataset: e.g., '/mavros/imu/data' or 'trisonica'
        - Partial path: If it uniquely identifies a dataset (e.g., 'imu/data')
        - List of paths: e.g., ['mavros/imu/data', 'trisonica'] returns dictionary
    
    Returns:
    --------
    pd.DataFrame or dict of pd.DataFrame
        If key is a string (except 'all_keys'): DataFrame containing the data from the specified dataset
        If key == 'all_keys' or key is a list: Dictionary mapping dataset paths to DataFrames
    '''
    
    f = h5py.File(filename, 'r')
    
    # Find all datasets
    all_datasets = find_all_datasets(filename)
    
    if not all_datasets:
        f.close()
        raise ValueError("No datasets found in the HDF5 file")
    
    # Handle list of keys
    if isinstance(key, list):
        print(f'Loading {len(key)} datasets...')
        result = {}
        for k in key:
            # Find the dataset path for this key
            if k in all_datasets:
                dataset_path = k
            else:
                # Try to find partial match
                matches = [ds for ds in all_datasets if k in ds]
                if len(matches) == 0:
                    f.close()
                    raise ValueError(f"No dataset found matching '{k}'. Available datasets:\n" + 
                                   '\n'.join([f"  {ds}" for ds in all_datasets]))
                elif len(matches) > 1:
                    f.close()
                    raise ValueError(f"Multiple datasets match '{k}':\n" + 
                                   '\n'.join([f"  {ds}" for ds in matches]) +
                                   "\nPlease be more specific.")
                else:
                    dataset_path = matches[0]
            
            print(f'  Loading: {dataset_path}')
            dataset = f[dataset_path]
            data = dataset[()]
            
            # Convert structured array to dictionary for DataFrame
            dic = {}
            for column_label in data.dtype.names:
                dic[column_label] = data[column_label]
            
            result[dataset_path] = pd.DataFrame(dic)
        
        f.close()
        print('Done loading datasets.')
        return result
    
    # Handle 'all_keys' option
    if key == 'all_keys':
        print(f'Loading all {len(all_datasets)} datasets...')
        result = {}
        for dataset_path in all_datasets:
            print(f'  Loading: {dataset_path}')
            dataset = f[dataset_path]
            data = dataset[()]
            
            # Convert structured array to dictionary for DataFrame
            dic = {}
            for column_label in data.dtype.names:
                dic[column_label] = data[column_label]
            
            result[dataset_path] = pd.DataFrame(dic)
        
        f.close()
        print('Done loading all datasets.')
        return result
    
    # Determine which dataset to load (single key)
    if key == 'first_key':
        if len(all_datasets) > 1:
            print('All available datasets:')
            for ds in all_datasets:
                print(f"  {ds}")
            print(f'\nWARNING: Loading first dataset only: {all_datasets[0]}')
        dataset_path = all_datasets[0]
    else:
        # Try to find the dataset
        # First try exact match
        if key in all_datasets:
            dataset_path = key
        else:
            # Try to find partial match
            matches = [ds for ds in all_datasets if key in ds]
            if len(matches) == 0:
                f.close()
                raise ValueError(f"No dataset found matching '{key}'. Available datasets:\n" + 
                               '\n'.join([f"  {ds}" for ds in all_datasets]))
            elif len(matches) > 1:
                f.close()
                raise ValueError(f"Multiple datasets match '{key}':\n" + 
                               '\n'.join([f"  {ds}" for ds in matches]) +
                               "\nPlease be more specific.")
            else:
                dataset_path = matches[0]
                print(f"Loading dataset: {dataset_path}")
    
    # Load the dataset
    dataset = f[dataset_path]
    if not isinstance(dataset, h5py.Dataset):
        f.close()
        raise TypeError(f"'{dataset_path}' is not a dataset")
    
    data = dataset[()]
    
    # Convert structured array to dictionary for DataFrame
    dic = {}
    for column_label in data.dtype.names:
        dic[column_label] = data[column_label]
    
    df = pd.DataFrame(dic)
    f.close()
    
    return df


def save_all_datasets_to_hdf5(input_filename, output_filename, compression='zlib', compression_opts=9):
    '''
    Load all datasets from a bag2hdf5 file and save them to a new HDF5 file
    with each dataset stored as a group containing its DataFrame.
    
    Parameters:
    -----------
    input_filename : str
        Path to the input HDF5 file (from bag2hdf5)
    output_filename : str
        Path to the output HDF5 file
    compression : str
        Compression algorithm (default: 'zlib')
        Options: 'zlib', 'lzo', 'bzip2', 'blosc', 'blosc2', etc.
        See pandas.HDFStore documentation for full list
    compression_opts : int
        Compression level 0-9 (default: 9 for maximum compression)
    
    Returns:
    --------
    dict
        Dictionary mapping dataset paths to DataFrames (same as get_pandas_dataframe_from_uncooperative_hdf5 with key='all_keys')
    '''
    import os
    import keyword
    
    def convert_bytes_to_str(df):
        """Convert byte string columns to regular string columns"""
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                # Check if it's a bytes column
                if len(df_copy[col]) > 0 and isinstance(df_copy[col].iloc[0], bytes):
                    df_copy[col] = df_copy[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        return df_copy
    
    def sanitize_path(path):
        """Sanitize path to avoid Python keywords and problematic characters"""
        # Split path into parts
        parts = path.split('/')
        sanitized_parts = []
        
        for part in parts:
            if not part:  # Skip empty parts
                continue
            
            # Replace spaces and other problematic characters
            clean_part = part.replace(' ', '_').replace('-', '_')
            
            # Check if it's a Python keyword and append underscore if so
            if keyword.iskeyword(clean_part):
                clean_part = clean_part + '_'
            
            sanitized_parts.append(clean_part)
        
        # Reconstruct path with leading slash
        return '/' + '/'.join(sanitized_parts)
    
    # Load all datasets
    print(f'Loading all datasets from {input_filename}...')
    all_dataframes = get_pandas_dataframe_from_uncooperative_hdf5(input_filename, key='all_keys')
    
    # Check if output file exists
    if os.path.exists(output_filename):
        print(f'WARNING: Output file {output_filename} already exists and will be overwritten.')
    
    # Save to new HDF5 file
    print(f'\nSaving to {output_filename}...')
    with pd.HDFStore(output_filename, mode='w', complevel=compression_opts, complib=compression) as store:
        for dataset_path, df in all_dataframes.items():
            # Sanitize the path
            clean_path = sanitize_path(dataset_path)
            
            # Convert byte strings to regular strings
            df_converted = convert_bytes_to_str(df)
            
            if clean_path != '/' + dataset_path.lstrip('/'):
                print(f'  Saving: {dataset_path} -> {clean_path} (shape: {df_converted.shape})')
            else:
                print(f'  Saving: {clean_path} (shape: {df_converted.shape})')
            
            store.put(clean_path, df_converted, format='table')
    
    print(f'\nSuccessfully saved {len(all_dataframes)} datasets to {output_filename}')
    return all_dataframes

def list_hdf5_keys(filename, print_keys=False):
    '''
    List all keys (dataset paths) in an HDF5 file.
    Works with both pandas HDFStore files and raw HDF5 files.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    
    Returns:
    --------
    list of str
        List of all keys/dataset paths in the file
    '''
    try:
        # Try pandas HDFStore first (for files created by save_all_datasets_to_hdf5)
        with pd.HDFStore(filename, 'r') as store:
            keys = store.keys()
            if print_keys:
                print(f"Keys in {filename}:")
                for key in keys:
                    print(f"  {key}")
            return list(keys)
    except:
        # Fall back to raw HDF5 (for files created by bag2hdf5)
        print(f"Keys in {filename}:")
        datasets = find_all_datasets(filename)
        for ds in datasets:
            if print_keys:
                print(f"  /{ds}" if not ds.startswith('/') else f"  {ds}")
        return datasets
