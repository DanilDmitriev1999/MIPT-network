import hydra
from omegaconf import DictConfig
from summarization_utils.TextRank import TextRank
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

class Bot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_for_summary = TextRank(cfg.Model.model_name)
        self.updater = Updater(cfg.Info.TOKEN, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.dispatcher.add_handler(CommandHandler("start", self.start))
        self.dispatcher.add_handler(CommandHandler("help", self.help))
        
        self.dispatcher.add_handler(MessageHandler(Filters.text, self.text))
        
        self.dispatcher.add_error_handler(self.error)
        
        self.updater.start_polling()
        
        self.updater.idle()
        
    def start(self, update, contex):
        update.message.replay_text('This simple bot knows how to summarize the text you enter. Just enter the text you want to shorten')

    def help(self, update, contex):
        update.message.replay_text('This simple bot knows how to summarize the text you enter. Just enter the text you want to shorten')

    def error(self, update, contex):
        update.message.reply_text('Input text for summarization')

    def text(self, update, contex):
        text_received = update.message.text
        summary = self.model_for_summary.get_summary(text_received)
        update.message.reply_text(f'You summary: \n {summary}')
        

@hydra.main(config_path="./config/", config_name="config")
def main(cfg: DictConfig):
    bot = Bot(cfg)


if __name__ == '__main__':
    main()
    