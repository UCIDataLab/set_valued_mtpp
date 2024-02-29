import torch
from .ppmodel import PPModel
from .set_assumption import *
from .decoder import HawkesDecoder, RMTPPDecoder
from .poisson import PoissonStaticCI
from .time_embedding import TemporalEmbedding

def get_model(
    channel_embedding_size,
    num_channels,
    dec_recurrent_hidden_size,
    set_assumption,
    condition_on_history,
    neural_hawkes=False,
    rmtpp=False,
    poisson=False,
    dyn_dom_buffer=16,
    num_layer_item=1,
):

    channel_embedding = torch.nn.Embedding(
        num_embeddings=num_channels,
        embedding_dim=channel_embedding_size
    )
    if poisson:
        return PoissonStaticCI(
            decoder=None,
            num_channels=num_channels,
            channel_embedding=channel_embedding,
            dyn_dom_buffer=dyn_dom_buffer,
        )
    elif rmtpp:
        decoder = RMTPPDecoder(
            channel_embedding=channel_embedding,
            time_embedding=TemporalEmbedding(
                embedding_dim=1,
                use_raw_time=False,
                use_delta_time=True,
                learnable_delta_weights=False,
                max_period=0,
            ),
            recurrent_hidden_size=dec_recurrent_hidden_size,
        )
    elif neural_hawkes:
        decoder = HawkesDecoder(
            channel_embedding=channel_embedding,
            time_embedding=TemporalEmbedding(
                embedding_dim=1,
                use_raw_time=False,
                use_delta_time=True,
                learnable_delta_weights=False,
                max_period=0,
            ),
            recurrent_hidden_size=dec_recurrent_hidden_size,
        )
    else:
        raise NotImplementedError
        
    if set_assumption == 'CI':
        if condition_on_history:
            return DynamicCondIndepSets(
                decoder=decoder,
                num_channels=num_channels,
                channel_embedding=channel_embedding,
                dyn_dom_buffer=dyn_dom_buffer,
                num_layer_item=num_layer_item,
            )
        else:
            return StaticCondIndepSets(
                decoder=decoder,
                num_channels=num_channels,
                channel_embedding=channel_embedding,
                dyn_dom_buffer=dyn_dom_buffer,
            )
    elif set_assumption == 'DPP':
        if condition_on_history:
            return DynamicDPPSets(
                decoder=decoder,
                num_channels=num_channels,
                channel_embedding=channel_embedding,
                dyn_dom_buffer=dyn_dom_buffer,
            )
        else:
            return StaticDPPSets(
                decoder=decoder,
                num_channels=num_channels,
                channel_embedding=channel_embedding,
                dyn_dom_buffer=dyn_dom_buffer,
            )
