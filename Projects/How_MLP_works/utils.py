
import pickle
from pathlib import Path

def save_model(model,
               model_name,
               save_dir):
    """
    Save model by pickle file to save_dir for model reusablity.

    Args: 
        model: Target model to save.
        model_name: Target name model will be saved.
            model_name should ends with 'pickle'
        save_dir: Target directory where model to saved.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(exist_ok=True, parents=True)

    assert model_name.endswith('.pickle'), "model_name should ends with '.pickle'"

    with open(save_dir / model_name, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model file saved in {save_dir/model_name}")

def load_model(model_name,
               saved_dir):
    """
    Load model from saved directory.

    Args:
        model_name: Model name to load from directory.
            Model name should ends with 'pickle'
        saved_dir: Directory where the saved model is in.

    Returns:
        model: 
    """
    saved_dir = Path(saved_dir)

    assert model_name.endswith('.pickle'), "model_name should ends with '.pickle'"

    with open(saved_dir / model_name, 'rb') as f:
        model = pickle.load(f)

    print(f"Model successsfully loaded!")
    return model
