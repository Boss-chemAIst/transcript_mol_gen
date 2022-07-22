from tensorflow import keras


""" Pre-trained model uploading """
model = keras.models.load_model('C:/projects/laboratory/Kapustina/transcript_mol_gen/transcriptome/trained_VAEs'
                                '/encoder_onehidden_vae.hdf5')


def transform_to_latent_representation(transcriptome_array):

    """

    transcriptome_array: ndarray of transcriptome data of shape (S, N), where S is transcriptome length and N is the
    number of transcriptomes.

    """

    return model.predict(transcriptome_array)
