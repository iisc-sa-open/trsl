import logging
import trsl
import node

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
entropy_reduction = []
trsl_instance = trsl.Trsl()
trsl_instance.train()

def avg_entropy_reduction():

    global trsl_instance

    traverse_node(trsl_instance.root)
    avg_entropy_reduction = (
        float(reduce(lambda x,y: x + y, entropy_reduction)) / len(entropy_reduction)
    )
    logging.info("Average Entropy Reduction: %s %%"
        % (
            avg_entropy_reduction
            )
    )
    logging.info("Maximum Reduction: %s %%"
        % (
            max(entropy_reduction)
            )
    )
    logging.info("Minimum Reduction: %s %%"
        % (
            min(entropy_reduction)
            )
    )

def traverse_node(temp):
    
    global trsl_instance
    if temp.rchild is None:
        entropy_reduction.append(
            100 * float(trsl_instance.root.entropy - temp.entropy) / trsl_instance.root.entropy
        )
        return
    traverse_node(temp.lchild)
    traverse_node(temp.rchild)


avg_entropy_reduction()